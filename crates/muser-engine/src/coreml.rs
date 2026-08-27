//! Serialized public Core ML inference constrained to CPU + Neural Engine.

use std::ffi::{c_void, CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use half::slice::HalfFloatSliceExt;
use objc::runtime::{Class, Object};
use objc::{msg_send, sel, sel_impl};

pub struct CoreMlModel {
    profile_label: String,
    model: *mut Object,
    input: *mut Object,
    provider: *mut Object,
    output_key: *mut Object,
    output: *mut Object,
    options: *mut Object,
    input_layout: MultiArrayLayout,
    output_layout: MultiArrayLayout,
    fallback_output_layout: OnceLock<MultiArrayLayout>,
    input_count: usize,
    prediction_lock: Mutex<()>,
}

/// Fixed named input declared by a public CoreML stateful package.
pub struct CoreMlTensorSpec<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
    pub data_type: CoreMlTensorDataType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoreMlTensorDataType {
    Float16,
    Float32,
}

impl CoreMlTensorDataType {
    fn raw_value(self) -> i64 {
        match self {
            Self::Float16 => 65552,
            Self::Float32 => 65568,
        }
    }
}

/// One prediction input. Names and shapes are checked against the loaded
/// package before any resident backing buffer is changed.
pub struct CoreMlTensorInput<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
    pub values: &'a [f32],
}

struct NamedMultiArray {
    name: String,
    array: *mut Object,
    layout: MultiArrayLayout,
    data_type: CoreMlTensorDataType,
}

/// Serialized public-CoreML prediction with multiple fixed inputs and one
/// persistent `MLState`. This uses only the macOS 15 public API:
/// `newState` and `predictionFromFeatures:usingState:options:error:`.
pub struct CoreMlStatefulModel {
    profile_label: String,
    model: *mut Object,
    inputs: Vec<NamedMultiArray>,
    provider: *mut Object,
    state: Mutex<*mut Object>,
    output_key: *mut Object,
    output: *mut Object,
    options: *mut Object,
    output_layout: MultiArrayLayout,
    fallback_output_layout: OnceLock<MultiArrayLayout>,
}

static PROFILE_CALL: AtomicUsize = AtomicUsize::new(0);

unsafe impl Send for CoreMlModel {}
unsafe impl Sync for CoreMlModel {}
unsafe impl Send for CoreMlStatefulModel {}
unsafe impl Sync for CoreMlStatefulModel {}

impl CoreMlModel {
    /// Load `.mlpackage` or `.mlmodelc` using only public Core ML APIs and
    /// force `MLComputeUnitsCPUAndNeuralEngine` (raw enum value 3).
    pub fn load(
        path: &Path,
        input_name: &str,
        output_name: &str,
        input_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<Self, String> {
        let input_name = CString::new(input_name).map_err(|_| "CoreML input name contains NUL")?;
        let output_name =
            CString::new(output_name).map_err(|_| "CoreML output name contains NUL")?;
        let input_count = input_shape
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .ok_or("CoreML shape overflow")?;
        if input_shape.is_empty()
            || input_shape.contains(&0)
            || output_shape.is_empty()
            || output_shape.contains(&0)
        {
            return Err("CoreML input/output shape is empty".into());
        }
        unsafe {
            let nsstring = Class::get("NSString").ok_or("NSString unavailable")?;
            let cpath = CString::new(path.to_str().ok_or("CoreML path is not UTF-8")?)
                .map_err(|_| "CoreML path contains NUL")?;
            let string: *mut Object = msg_send![nsstring,stringWithUTF8String:cpath.as_ptr()];
            let nsurl = Class::get("NSURL").ok_or("NSURL unavailable")?;
            let source: *mut Object = msg_send![nsurl,fileURLWithPath:string];
            if source.is_null() {
                return Err("failed to create CoreML URL".into());
            }
            let model_class = Class::get("MLModel").ok_or("MLModel unavailable")?;
            let compiled = if path.extension().and_then(|x| x.to_str()) == Some("mlmodelc") {
                source
            } else {
                let mut error: *mut Object = std::ptr::null_mut();
                let url: *mut Object =
                    msg_send![model_class,compileModelAtURL:source error:&mut error];
                if url.is_null() || !error.is_null() {
                    return Err(objc_error("CoreML compile failed", error));
                }
                url
            };
            let configuration_class =
                Class::get("MLModelConfiguration").ok_or("MLModelConfiguration unavailable")?;
            let configuration: *mut Object = msg_send![configuration_class, new];
            if configuration.is_null() {
                return Err("failed to allocate MLModelConfiguration".into());
            }
            let _: () = msg_send![configuration,setComputeUnits:3i64];
            let mut error: *mut Object = std::ptr::null_mut();
            let model: *mut Object = msg_send![model_class,modelWithContentsOfURL:compiled configuration:configuration error:&mut error];
            let _: () = msg_send![configuration, release];
            if model.is_null() || !error.is_null() {
                return Err(objc_error("CoreML model load failed", error));
            }
            let _: *mut Object = msg_send![model, retain];

            // Every release shard has a fixed input shape. Allocate its public
            // MLMultiArray and feature provider once, then mutate only the
            // backing data under `prediction_lock`. Rebuilding this object
            // graph for every one of the 70 shard calls dominated host tax.
            let number = Class::get("NSNumber").ok_or("NSNumber unavailable")?;
            let array = Class::get("NSArray").ok_or("NSArray unavailable")?;
            let dims = input_shape
                .iter()
                .map(|&dim| {
                    let dim = i64::try_from(dim).map_err(|_| "CoreML dimension exceeds i64")?;
                    Ok(msg_send![number,numberWithLongLong:dim])
                })
                .collect::<Result<Vec<*mut Object>, String>>()?;
            let shape_obj: *mut Object =
                msg_send![array,arrayWithObjects:dims.as_ptr() count:dims.len()];
            let multi = Class::get("MLMultiArray").ok_or("MLMultiArray unavailable")?;
            let mut input_error: *mut Object = std::ptr::null_mut();
            let input: *mut Object = msg_send![multi, alloc];
            let input: *mut Object =
                msg_send![input,initWithShape:shape_obj dataType:65568i64 error:&mut input_error];
            if input.is_null() || !input_error.is_null() {
                let _: () = msg_send![model, release];
                return Err(objc_error("CoreML input allocation failed", input_error));
            }
            let input_strides = match multi_array_usizes(input, "strides") {
                Ok(strides) => strides,
                Err(error) => {
                    let _: () = msg_send![input, release];
                    let _: () = msg_send![model, release];
                    return Err(error);
                }
            };
            if input_strides.len() != input_shape.len() {
                let _: () = msg_send![input, release];
                let _: () = msg_send![model, release];
                return Err("CoreML input stride rank differs from its shape".into());
            }
            let input_layout = match MultiArrayLayout::new(input_shape.to_vec(), input_strides) {
                Ok(layout) => layout,
                Err(error) => {
                    let _: () = msg_send![input, release];
                    let _: () = msg_send![model, release];
                    return Err(error);
                }
            };
            let string = Class::get("NSString").ok_or("NSString unavailable")?;
            let input_key: *mut Object = msg_send![string,stringWithUTF8String:input_name.as_ptr()];
            let output_key: *mut Object =
                msg_send![string,stringWithUTF8String:output_name.as_ptr()];
            let _: *mut Object = msg_send![output_key, retain];
            let feature = Class::get("MLFeatureValue").ok_or("MLFeatureValue unavailable")?;
            let feature: *mut Object = msg_send![feature,featureValueWithMultiArray:input];
            let dictionary_class = Class::get("NSDictionary").ok_or("NSDictionary unavailable")?;
            let input_dictionary: *mut Object =
                msg_send![dictionary_class,dictionaryWithObject:feature forKey:input_key];
            let provider_class = Class::get("MLDictionaryFeatureProvider")
                .ok_or("MLDictionaryFeatureProvider unavailable")?;
            let mut provider_error: *mut Object = std::ptr::null_mut();
            let provider: *mut Object = msg_send![provider_class, alloc];
            let provider: *mut Object =
                msg_send![provider,initWithDictionary:input_dictionary error:&mut provider_error];
            if provider.is_null() || !provider_error.is_null() {
                let _: () = msg_send![output_key, release];
                let _: () = msg_send![input, release];
                let _: () = msg_send![model, release];
                return Err(objc_error("CoreML feature provider failed", provider_error));
            }

            // Ask Core ML to write directly into one resident MLMultiArray.
            // This is the public `MLPredictionOptions.outputBackings` path;
            // Core ML may decline it, in which case `predict_into` validates
            // and reads the framework-owned fallback result.
            let output_dims = output_shape
                .iter()
                .map(|&dim| {
                    let dim = i64::try_from(dim).map_err(|_| "CoreML dimension exceeds i64")?;
                    Ok(msg_send![number,numberWithLongLong:dim])
                })
                .collect::<Result<Vec<*mut Object>, String>>()?;
            let output_shape_obj: *mut Object =
                msg_send![array,arrayWithObjects:output_dims.as_ptr() count:output_dims.len()];
            let mut output_error: *mut Object = std::ptr::null_mut();
            let output: *mut Object = msg_send![multi, alloc];
            let output: *mut Object = msg_send![output,initWithShape:output_shape_obj dataType:65568i64 error:&mut output_error];
            if output.is_null() || !output_error.is_null() {
                let _: () = msg_send![provider, release];
                let _: () = msg_send![output_key, release];
                let _: () = msg_send![input, release];
                let _: () = msg_send![model, release];
                return Err(objc_error(
                    "CoreML output backing allocation failed",
                    output_error,
                ));
            }
            let output_layout = match MultiArrayLayout::new(
                output_shape.to_vec(),
                multi_array_usizes(output, "strides")?,
            ) {
                Ok(layout) => layout,
                Err(error) => {
                    let _: () = msg_send![output, release];
                    let _: () = msg_send![provider, release];
                    let _: () = msg_send![output_key, release];
                    let _: () = msg_send![input, release];
                    let _: () = msg_send![model, release];
                    return Err(error);
                }
            };
            let output_backings: *mut Object =
                msg_send![dictionary_class,dictionaryWithObject:output forKey:output_key];
            let options_class =
                Class::get("MLPredictionOptions").ok_or("MLPredictionOptions unavailable")?;
            let options: *mut Object = msg_send![options_class, new];
            if options.is_null() {
                let _: () = msg_send![output, release];
                let _: () = msg_send![provider, release];
                let _: () = msg_send![output_key, release];
                let _: () = msg_send![input, release];
                let _: () = msg_send![model, release];
                return Err("failed to allocate MLPredictionOptions".into());
            }
            let _: () = msg_send![options,setOutputBackings:output_backings];
            Ok(Self {
                profile_label: path.display().to_string(),
                model,
                input,
                provider,
                output_key,
                output,
                options,
                input_layout,
                output_layout,
                fallback_output_layout: OnceLock::new(),
                input_count,
                prediction_lock: Mutex::new(()),
            })
        }
    }

    pub fn predict_into(
        &self,
        input: &[f32],
        shape: &[usize],
        output: &mut [f32],
    ) -> Result<(), String> {
        let _guard = self
            .prediction_lock
            .lock()
            .map_err(|_| "CoreML prediction lock poisoned")?;
        if shape != self.input_layout.shape || input.len() != self.input_count {
            return Err("CoreML input shape/data mismatch".into());
        }
        let profile = std::env::var_os("MUSER_COREML_CALL_PROFILE").is_some();
        let started = Instant::now();
        unsafe {
            let ptr: *mut c_void = msg_send![self.input, dataPointer];
            if ptr.is_null() && self.input_count > 0 {
                return Err("CoreML input data pointer is null".into());
            }
            copy_to_multi_array(input, ptr as *mut f32, &self.input_layout);
            let input_done = started.elapsed();
            let mut prediction_error: *mut Object = std::ptr::null_mut();
            let result: *mut Object = msg_send![self.model,predictionFromFeatures:self.provider options:self.options error:&mut prediction_error];
            let prediction_done = started.elapsed();
            if result.is_null() || !prediction_error.is_null() {
                return Err(objc_error("CoreML prediction failed", prediction_error));
            }
            let out_feature: *mut Object = msg_send![result,featureValueForName:self.output_key];
            let output_array: *mut Object = msg_send![out_feature, multiArrayValue];
            if output_array.is_null() {
                return Err("CoreML output is absent".into());
            }
            let output_layout = if output_array == self.output {
                &self.output_layout
            } else if let Some(layout) = self.fallback_output_layout.get() {
                if !multi_array_matches(output_array, "shape", &layout.shape)?
                    || !multi_array_matches(output_array, "strides", &layout.strides)?
                {
                    return Err("CoreML output layout changed between predictions".into());
                }
                layout
            } else {
                let layout = MultiArrayLayout::new(
                    multi_array_usizes(output_array, "shape")?,
                    multi_array_usizes(output_array, "strides")?,
                )?;
                let _ = self.fallback_output_layout.set(layout);
                self.fallback_output_layout
                    .get()
                    .ok_or("CoreML output layout was not retained")?
            };
            if output_layout.elements != output.len() {
                return Err("CoreML output shape differs from its declared geometry".into());
            }
            let data: *const c_void = msg_send![output_array, dataPointer];
            let dtype: i64 = msg_send![output_array, dataType];
            if data.is_null() {
                return Err("CoreML output data pointer is null".into());
            }
            match dtype {
                65568 => {
                    copy_from_multi_array(data as *const f32, output, output_layout, |value| value)
                }
                65552 => copy_from_f16_multi_array(data as *const u16, output, output_layout),
                65600 => {
                    copy_from_multi_array(data as *const f64, output, output_layout, |value| {
                        value as f32
                    })
                }
                _ => return Err(format!("unsupported CoreML output dtype {dtype}")),
            }
            if profile {
                let call = PROFILE_CALL.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "[muser-coreml-call] call={call} kind=stateless label={:?} input_elements={} output_elements={} input_ns={} predict_ns={} output_ns={} direct_output_backing={}",
                    self.profile_label,
                    input.len(),
                    output.len(),
                    input_done.as_nanos(),
                    prediction_done.saturating_sub(input_done).as_nanos(),
                    started.elapsed().saturating_sub(prediction_done).as_nanos(),
                    output_array == self.output,
                );
            }
            Ok(())
        }
    }
}

impl CoreMlStatefulModel {
    pub fn load(
        path: &Path,
        input_specs: &[CoreMlTensorSpec<'_>],
        output_name: &str,
        output_shape: &[usize],
        output_data_type: CoreMlTensorDataType,
    ) -> Result<Self, String> {
        if input_specs.is_empty() {
            return Err("stateful CoreML model has no inputs".into());
        }
        if output_shape.is_empty() || output_shape.contains(&0) {
            return Err("stateful CoreML output shape is empty".into());
        }
        let mut unique = std::collections::BTreeSet::new();
        for spec in input_specs {
            if spec.name.is_empty()
                || !unique.insert(spec.name)
                || spec.shape.is_empty()
                || spec.shape.contains(&0)
            {
                return Err("stateful CoreML input specification is invalid".into());
            }
        }
        let output_name =
            CString::new(output_name).map_err(|_| "CoreML output name contains NUL")?;
        unsafe {
            let model = load_public_model(path)?;
            let mut inputs = Vec::with_capacity(input_specs.len());
            let mut keys = Vec::with_capacity(input_specs.len());
            let mut values = Vec::with_capacity(input_specs.len());
            let string = Class::get("NSString").ok_or("NSString unavailable")?;
            let feature = Class::get("MLFeatureValue").ok_or("MLFeatureValue unavailable")?;
            for spec in input_specs {
                let name = CString::new(spec.name)
                    .map_err(|_| "stateful CoreML input name contains NUL")?;
                let array = allocate_typed_multi_array(spec.shape, spec.data_type)?;
                let layout = MultiArrayLayout::new(
                    spec.shape.to_vec(),
                    multi_array_usizes(array, "strides")?,
                )?;
                let key: *mut Object = msg_send![string,stringWithUTF8String:name.as_ptr()];
                let value: *mut Object = msg_send![feature,featureValueWithMultiArray:array];
                keys.push(key);
                values.push(value);
                inputs.push(NamedMultiArray {
                    name: spec.name.to_owned(),
                    array,
                    layout,
                    data_type: spec.data_type,
                });
            }
            let dictionary = Class::get("NSDictionary").ok_or("NSDictionary unavailable")?;
            let input_dictionary: *mut Object = msg_send![dictionary,dictionaryWithObjects:values.as_ptr() forKeys:keys.as_ptr() count:keys.len()];
            let provider_class = Class::get("MLDictionaryFeatureProvider")
                .ok_or("MLDictionaryFeatureProvider unavailable")?;
            let mut provider_error: *mut Object = std::ptr::null_mut();
            let provider: *mut Object = msg_send![provider_class, alloc];
            let provider: *mut Object =
                msg_send![provider,initWithDictionary:input_dictionary error:&mut provider_error];
            if provider.is_null() || !provider_error.is_null() {
                release_named_inputs(&inputs);
                let _: () = msg_send![model, release];
                return Err(objc_error(
                    "stateful CoreML feature provider failed",
                    provider_error,
                ));
            }
            let state: *mut Object = msg_send![model, newState];
            if state.is_null() {
                let _: () = msg_send![provider, release];
                release_named_inputs(&inputs);
                let _: () = msg_send![model, release];
                return Err("stateful CoreML newState returned null".into());
            }
            let output_key: *mut Object =
                msg_send![string,stringWithUTF8String:output_name.as_ptr()];
            let _: *mut Object = msg_send![output_key, retain];
            let output = match allocate_typed_multi_array(output_shape, output_data_type) {
                Ok(output) => output,
                Err(error) => {
                    let _: () = msg_send![state, release];
                    let _: () = msg_send![provider, release];
                    release_named_inputs(&inputs);
                    let _: () = msg_send![output_key, release];
                    let _: () = msg_send![model, release];
                    return Err(error);
                }
            };
            let output_layout = MultiArrayLayout::new(
                output_shape.to_vec(),
                multi_array_usizes(output, "strides")?,
            )?;
            let output_backings: *mut Object =
                msg_send![dictionary,dictionaryWithObject:output forKey:output_key];
            let options_class =
                Class::get("MLPredictionOptions").ok_or("MLPredictionOptions unavailable")?;
            let options: *mut Object = msg_send![options_class, new];
            if options.is_null() {
                let _: () = msg_send![output, release];
                let _: () = msg_send![state, release];
                let _: () = msg_send![provider, release];
                release_named_inputs(&inputs);
                let _: () = msg_send![output_key, release];
                let _: () = msg_send![model, release];
                return Err("failed to allocate stateful MLPredictionOptions".into());
            }
            let _: () = msg_send![options,setOutputBackings:output_backings];
            Ok(Self {
                profile_label: path.display().to_string(),
                model,
                inputs,
                provider,
                state: Mutex::new(state),
                output_key,
                output,
                options,
                output_layout,
                fallback_output_layout: OnceLock::new(),
            })
        }
    }

    pub fn reset_state(&self) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "CoreML state lock poisoned")?;
        unsafe {
            let replacement: *mut Object = msg_send![self.model, newState];
            if replacement.is_null() {
                return Err("stateful CoreML newState returned null".into());
            }
            let previous = std::mem::replace(&mut *state, replacement);
            let _: () = msg_send![previous, release];
        }
        Ok(())
    }

    pub fn predict_into(
        &self,
        inputs: &[CoreMlTensorInput<'_>],
        output: &mut [f32],
    ) -> Result<(), String> {
        let state = self
            .state
            .lock()
            .map_err(|_| "CoreML state lock poisoned")?;
        if inputs.len() != self.inputs.len() {
            return Err("stateful CoreML input count mismatch".into());
        }
        for (input, resident) in inputs.iter().zip(&self.inputs) {
            if input.name != resident.name
                || input.shape != resident.layout.shape
                || input.values.len() != resident.layout.elements
            {
                return Err(format!(
                    "stateful CoreML input {} shape/data mismatch",
                    input.name
                ));
            }
        }
        if output.len() != self.output_layout.elements {
            return Err("stateful CoreML output length mismatch".into());
        }
        let profile = std::env::var_os("MUSER_COREML_CALL_PROFILE").is_some();
        let started = Instant::now();
        unsafe {
            for (input, resident) in inputs.iter().zip(&self.inputs) {
                let pointer: *mut c_void = msg_send![resident.array, dataPointer];
                if pointer.is_null() {
                    return Err(format!(
                        "stateful CoreML input {} data pointer is null",
                        input.name
                    ));
                }
                match resident.data_type {
                    CoreMlTensorDataType::Float16 => {
                        copy_to_f16_multi_array(input.values, pointer as *mut u16, &resident.layout)
                    }
                    CoreMlTensorDataType::Float32 => {
                        copy_to_multi_array(input.values, pointer as *mut f32, &resident.layout)
                    }
                }
            }
            let input_done = started.elapsed();
            let mut error: *mut Object = std::ptr::null_mut();
            let result: *mut Object = msg_send![self.model,predictionFromFeatures:self.provider usingState:*state options:self.options error:&mut error];
            let prediction_done = started.elapsed();
            if result.is_null() || !error.is_null() {
                return Err(objc_error("stateful CoreML prediction failed", error));
            }
            let feature: *mut Object = msg_send![result,featureValueForName:self.output_key];
            let array: *mut Object = msg_send![feature, multiArrayValue];
            if array.is_null() {
                return Err("stateful CoreML output is absent".into());
            }
            let layout = if array == self.output {
                &self.output_layout
            } else if let Some(layout) = self.fallback_output_layout.get() {
                if !multi_array_matches(array, "shape", &layout.shape)?
                    || !multi_array_matches(array, "strides", &layout.strides)?
                {
                    return Err("stateful CoreML output layout changed".into());
                }
                layout
            } else {
                let layout = MultiArrayLayout::new(
                    multi_array_usizes(array, "shape")?,
                    multi_array_usizes(array, "strides")?,
                )?;
                let _ = self.fallback_output_layout.set(layout);
                self.fallback_output_layout
                    .get()
                    .ok_or("stateful CoreML output layout was not retained")?
            };
            if layout.elements != output.len() {
                return Err("stateful CoreML output geometry differs".into());
            }
            copy_coreml_output(array, output, layout)?;
            if profile {
                let call = PROFILE_CALL.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "[muser-coreml-call] call={call} kind=stateful label={:?} input_elements={} output_elements={} input_ns={} predict_ns={} output_ns={} direct_output_backing={}",
                    self.profile_label,
                    inputs.iter().map(|input| input.values.len()).sum::<usize>(),
                    output.len(),
                    input_done.as_nanos(),
                    prediction_done.saturating_sub(input_done).as_nanos(),
                    started.elapsed().saturating_sub(prediction_done).as_nanos(),
                    array == self.output,
                );
            }
            Ok(())
        }
    }
}

unsafe fn load_public_model(path: &Path) -> Result<*mut Object, String> {
    let string_class = Class::get("NSString").ok_or("NSString unavailable")?;
    let cpath = CString::new(path.to_str().ok_or("CoreML path is not UTF-8")?)
        .map_err(|_| "CoreML path contains NUL")?;
    let path_string: *mut Object = msg_send![string_class,stringWithUTF8String:cpath.as_ptr()];
    let url_class = Class::get("NSURL").ok_or("NSURL unavailable")?;
    let source: *mut Object = msg_send![url_class,fileURLWithPath:path_string];
    if source.is_null() {
        return Err("failed to create CoreML URL".into());
    }
    let model_class = Class::get("MLModel").ok_or("MLModel unavailable")?;
    let compiled = if path.extension().and_then(|value| value.to_str()) == Some("mlmodelc") {
        source
    } else {
        let mut error: *mut Object = std::ptr::null_mut();
        let result: *mut Object = msg_send![model_class,compileModelAtURL:source error:&mut error];
        if result.is_null() || !error.is_null() {
            return Err(objc_error("CoreML compile failed", error));
        }
        result
    };
    let configuration_class =
        Class::get("MLModelConfiguration").ok_or("MLModelConfiguration unavailable")?;
    let configuration: *mut Object = msg_send![configuration_class, new];
    if configuration.is_null() {
        return Err("failed to allocate MLModelConfiguration".into());
    }
    let _: () = msg_send![configuration,setComputeUnits:3i64];
    let mut error: *mut Object = std::ptr::null_mut();
    let model: *mut Object = msg_send![model_class,modelWithContentsOfURL:compiled configuration:configuration error:&mut error];
    let _: () = msg_send![configuration, release];
    if model.is_null() || !error.is_null() {
        return Err(objc_error("CoreML model load failed", error));
    }
    let _: *mut Object = msg_send![model, retain];
    Ok(model)
}

unsafe fn allocate_multi_array(shape: &[usize]) -> Result<*mut Object, String> {
    allocate_typed_multi_array(shape, CoreMlTensorDataType::Float32)
}

unsafe fn allocate_typed_multi_array(
    shape: &[usize],
    data_type: CoreMlTensorDataType,
) -> Result<*mut Object, String> {
    if shape.is_empty() || shape.contains(&0) {
        return Err("CoreML array shape is empty".into());
    }
    let number = Class::get("NSNumber").ok_or("NSNumber unavailable")?;
    let array = Class::get("NSArray").ok_or("NSArray unavailable")?;
    let dimensions = shape
        .iter()
        .map(|&dimension| {
            let dimension = i64::try_from(dimension).map_err(|_| "CoreML dimension exceeds i64")?;
            Ok(msg_send![number,numberWithLongLong:dimension])
        })
        .collect::<Result<Vec<*mut Object>, String>>()?;
    let shape_object: *mut Object =
        msg_send![array,arrayWithObjects:dimensions.as_ptr() count:dimensions.len()];
    let multi = Class::get("MLMultiArray").ok_or("MLMultiArray unavailable")?;
    let mut error: *mut Object = std::ptr::null_mut();
    let result: *mut Object = msg_send![multi, alloc];
    let result: *mut Object = msg_send![result,initWithShape:shape_object dataType:data_type.raw_value() error:&mut error];
    if result.is_null() || !error.is_null() {
        return Err(objc_error("CoreML array allocation failed", error));
    }
    Ok(result)
}

unsafe fn release_named_inputs(inputs: &[NamedMultiArray]) {
    for input in inputs {
        if !input.array.is_null() {
            let _: () = msg_send![input.array, release];
        }
    }
}

unsafe fn copy_coreml_output(
    array: *mut Object,
    output: &mut [f32],
    layout: &MultiArrayLayout,
) -> Result<(), String> {
    let data: *const c_void = msg_send![array, dataPointer];
    let dtype: i64 = msg_send![array, dataType];
    if data.is_null() {
        return Err("CoreML output data pointer is null".into());
    }
    match dtype {
        65568 => copy_from_multi_array(data as *const f32, output, layout, |value| value),
        65552 => copy_from_f16_multi_array(data as *const u16, output, layout),
        65600 => copy_from_multi_array(data as *const f64, output, layout, |value| value as f32),
        _ => return Err(format!("unsupported CoreML output dtype {dtype}")),
    }
    Ok(())
}

struct MultiArrayLayout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    elements: usize,
    contiguous: bool,
}

impl MultiArrayLayout {
    fn new(shape: Vec<usize>, strides: Vec<usize>) -> Result<Self, String> {
        if shape.is_empty() || shape.contains(&0) || shape.len() != strides.len() {
            return Err("CoreML array shape/stride geometry is invalid".into());
        }
        let elements = shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or("CoreML array shape overflow")?;
        let contiguous = is_contiguous(&shape, &strides);
        Ok(Self {
            shape,
            strides,
            elements,
            contiguous,
        })
    }
}

fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expected = 1usize;
    for (&dimension, &stride) in shape.iter().zip(strides).rev() {
        if stride != expected {
            return false;
        }
        let Some(next) = expected.checked_mul(dimension) else {
            return false;
        };
        expected = next;
    }
    true
}

#[cfg(test)]
fn contiguous_strides(shape: &[usize]) -> Option<Vec<usize>> {
    let mut stride = 1usize;
    let mut strides = vec![0; shape.len()];
    for (index, &dimension) in shape.iter().enumerate().rev() {
        strides[index] = stride;
        stride = stride.checked_mul(dimension)?;
    }
    Some(strides)
}

fn physical_offset(mut logical: usize, shape: &[usize], strides: &[usize]) -> usize {
    let mut offset = 0usize;
    for (&dimension, &stride) in shape.iter().zip(strides).rev() {
        let coordinate = logical % dimension;
        logical /= dimension;
        offset += coordinate * stride;
    }
    offset
}

unsafe fn copy_to_multi_array(input: &[f32], destination: *mut f32, layout: &MultiArrayLayout) {
    if layout.contiguous {
        std::ptr::copy_nonoverlapping(input.as_ptr(), destination, input.len());
        return;
    }
    for (logical, &value) in input.iter().enumerate() {
        *destination.add(physical_offset(logical, &layout.shape, &layout.strides)) = value;
    }
}

unsafe fn copy_to_f16_multi_array(input: &[f32], destination: *mut u16, layout: &MultiArrayLayout) {
    if layout.contiguous {
        let destination =
            std::slice::from_raw_parts_mut(destination.cast::<half::f16>(), input.len());
        destination.convert_from_f32_slice(input);
        return;
    }
    for (logical, &value) in input.iter().enumerate() {
        *destination.add(physical_offset(logical, &layout.shape, &layout.strides)) =
            half::f16::from_f32(value).to_bits();
    }
}

unsafe fn copy_from_f16_multi_array(
    source: *const u16,
    output: &mut [f32],
    layout: &MultiArrayLayout,
) {
    if layout.contiguous {
        let source = std::slice::from_raw_parts(source.cast::<half::f16>(), output.len());
        source.convert_to_f32_slice(output);
        return;
    }
    for (logical, value) in output.iter_mut().enumerate() {
        *value = half::f16::from_bits(*source.add(physical_offset(
            logical,
            &layout.shape,
            &layout.strides,
        )))
        .to_f32();
    }
}

unsafe fn copy_from_multi_array<T: Copy>(
    source: *const T,
    output: &mut [f32],
    layout: &MultiArrayLayout,
    convert: impl Fn(T) -> f32,
) {
    if layout.contiguous {
        for (index, value) in output.iter_mut().enumerate() {
            *value = convert(*source.add(index));
        }
        return;
    }
    for (logical, value) in output.iter_mut().enumerate() {
        *value = convert(*source.add(physical_offset(logical, &layout.shape, &layout.strides)));
    }
}

unsafe fn multi_array_matches(
    array: *mut Object,
    property: &str,
    expected: &[usize],
) -> Result<bool, String> {
    let values: *mut Object = match property {
        "shape" => msg_send![array, shape],
        "strides" => msg_send![array, strides],
        _ => return Err(format!("unsupported CoreML array property {property}")),
    };
    if values.is_null() {
        return Err(format!("CoreML array {property} is absent"));
    }
    let count: usize = msg_send![values, count];
    if count != expected.len() {
        return Ok(false);
    }
    for (index, &expected_value) in expected.iter().enumerate() {
        let number: *mut Object = msg_send![values, objectAtIndex:index];
        let value: u64 = msg_send![number, unsignedLongLongValue];
        if usize::try_from(value).ok() != Some(expected_value) {
            return Ok(false);
        }
    }
    Ok(true)
}

unsafe fn multi_array_usizes(array: *mut Object, property: &str) -> Result<Vec<usize>, String> {
    let values: *mut Object = match property {
        "shape" => msg_send![array, shape],
        "strides" => msg_send![array, strides],
        _ => return Err(format!("unsupported CoreML array property {property}")),
    };
    if values.is_null() {
        return Err(format!("CoreML array {property} is absent"));
    }
    let count: usize = msg_send![values, count];
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let number: *mut Object = msg_send![values, objectAtIndex:index];
        let value: u64 = msg_send![number, unsignedLongLongValue];
        result.push(
            usize::try_from(value).map_err(|_| format!("CoreML array {property} exceeds usize"))?,
        );
    }
    Ok(result)
}

impl Drop for CoreMlModel {
    fn drop(&mut self) {
        unsafe {
            if !self.provider.is_null() {
                let _: () = msg_send![self.provider, release];
            }
            if !self.input.is_null() {
                let _: () = msg_send![self.input, release];
            }
            if !self.output_key.is_null() {
                let _: () = msg_send![self.output_key, release];
            }
            if !self.options.is_null() {
                let _: () = msg_send![self.options, release];
            }
            if !self.output.is_null() {
                let _: () = msg_send![self.output, release];
            }
            if !self.model.is_null() {
                let _: () = msg_send![self.model, release];
            }
        }
    }
}

impl Drop for CoreMlStatefulModel {
    fn drop(&mut self) {
        unsafe {
            if let Ok(state) = self.state.get_mut() {
                if !(*state).is_null() {
                    let _: () = msg_send![*state, release];
                }
            }
            if !self.provider.is_null() {
                let _: () = msg_send![self.provider, release];
            }
            release_named_inputs(&self.inputs);
            if !self.output_key.is_null() {
                let _: () = msg_send![self.output_key, release];
            }
            if !self.options.is_null() {
                let _: () = msg_send![self.options, release];
            }
            if !self.output.is_null() {
                let _: () = msg_send![self.output, release];
            }
            if !self.model.is_null() {
                let _: () = msg_send![self.model, release];
            }
        }
    }
}

unsafe fn objc_error(prefix: &str, error: *mut Object) -> String {
    if error.is_null() {
        return format!("{prefix}: unknown error");
    }
    let description: *mut Object = msg_send![error, localizedDescription];
    if description.is_null() {
        return format!("{prefix}: missing description");
    }
    let bytes: *const i8 = msg_send![description, UTF8String];
    if bytes.is_null() {
        format!("{prefix}: invalid description")
    } else {
        format!("{prefix}: {}", CStr::from_ptr(bytes).to_string_lossy())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        contiguous_strides, copy_from_f16_multi_array, copy_to_f16_multi_array, physical_offset,
        MultiArrayLayout,
    };

    #[test]
    fn nctw_contiguous_offsets_match_channel_major_tokens() {
        let shape = [1, 3, 4, 1];
        let strides = contiguous_strides(&shape).unwrap();
        assert_eq!(strides, [12, 4, 1, 1]);
        assert_eq!(physical_offset(0, &shape, &strides), 0);
        assert_eq!(physical_offset(7, &shape, &strides), 7);
        assert_eq!(physical_offset(11, &shape, &strides), 11);
    }

    #[test]
    fn logical_flattening_respects_noncontiguous_multiarray_strides() {
        let shape = [1, 2, 3, 1];
        let strides = [32, 8, 2, 1];
        assert_eq!(physical_offset(0, &shape, &strides), 0);
        assert_eq!(physical_offset(1, &shape, &strides), 2);
        assert_eq!(physical_offset(2, &shape, &strides), 4);
        assert_eq!(physical_offset(3, &shape, &strides), 8);
        assert_eq!(physical_offset(5, &shape, &strides), 12);
    }

    #[test]
    fn contiguous_half_boundary_conversion_round_trips() {
        let input = [0.0, 1.0, -2.5, 0.333_251_95, 65_504.0];
        let layout = MultiArrayLayout::new(vec![1, 5, 1, 1], vec![5, 1, 1, 1]).unwrap();
        let mut bits = vec![0_u16; input.len()];
        unsafe { copy_to_f16_multi_array(&input, bits.as_mut_ptr(), &layout) };
        assert_eq!(
            bits,
            input
                .iter()
                .map(|&value| half::f16::from_f32(value).to_bits())
                .collect::<Vec<_>>()
        );
        let mut output = vec![0.0; input.len()];
        unsafe { copy_from_f16_multi_array(bits.as_ptr(), &mut output, &layout) };
        assert_eq!(
            output,
            input
                .iter()
                .map(|&value| half::f16::from_f32(value).to_f32())
                .collect::<Vec<_>>()
        );
    }
}
