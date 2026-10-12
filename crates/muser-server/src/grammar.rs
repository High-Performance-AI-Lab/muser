//! Pinned llama-style GBNF parsing and incremental UTF-8 matching.
//!
//! The matcher is an Earley recognizer over Unicode code points. It keeps all
//! ambiguous stacks alive, accepts token byte fragments that end in a partial
//! UTF-8 sequence, and exposes acceptance separately so EOS is legal only at
//! a completed root rule.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GrammarMatcher {
    grammar: Grammar,
    branches: Vec<MatchBranch>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct MatchBranch {
    chart: Vec<HashSet<Item>>,
    pending_utf8: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Grammar {
    productions: Vec<Production>,
    by_lhs: Vec<Vec<usize>>,
    start_production: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Production {
    lhs: usize,
    rhs: Vec<Symbol>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Symbol {
    Rule(usize),
    Character(CharSet),
    Token(TokenTerminal),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenTerminal {
    inverted: bool,
    value: TokenValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum TokenValue {
    Id(u32),
    Piece(Vec<u8>),
}

impl TokenTerminal {
    fn matches(&self, token: u32, bytes: &[u8]) -> bool {
        let equal = match &self.value {
            TokenValue::Id(expected) => token == *expected,
            TokenValue::Piece(expected) => bytes == expected,
        };
        equal != self.inverted
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CharSet {
    inverted: bool,
    ranges: Vec<(u32, u32)>,
}

impl CharSet {
    fn one(value: u32) -> Self {
        Self {
            inverted: false,
            ranges: vec![(value, value)],
        }
    }

    fn matches(&self, value: u32) -> bool {
        let included = self
            .ranges
            .iter()
            .any(|&(lower, upper)| lower <= value && value <= upper);
        included != self.inverted
    }

    fn intersects(&self, lower: u32, upper: u32) -> bool {
        if !self.inverted {
            return self
                .ranges
                .iter()
                .any(|&(range_lower, range_upper)| range_lower <= upper && lower <= range_upper);
        }

        let mut covered = self
            .ranges
            .iter()
            .filter_map(|&(range_lower, range_upper)| {
                let range_lower = range_lower.max(lower);
                let range_upper = range_upper.min(upper);
                (range_lower <= range_upper).then_some((range_lower, range_upper))
            })
            .collect::<Vec<_>>();
        covered.sort_unstable();
        let mut cursor = lower;
        for (range_lower, range_upper) in covered {
            if range_lower > cursor {
                return true;
            }
            cursor = cursor.max(range_upper.saturating_add(1));
            if cursor > upper {
                return false;
            }
        }
        cursor <= upper
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Item {
    production: usize,
    dot: usize,
    origin: usize,
}

impl GrammarMatcher {
    pub(crate) fn parse(source: &str, root: &str) -> Result<Self, String> {
        let grammar = Parser::new(source).parse(root)?;
        let mut branch = MatchBranch {
            chart: vec![HashSet::new()],
            pending_utf8: Vec::new(),
        };
        branch.chart[0].insert(Item {
            production: grammar.start_production,
            dot: 0,
            origin: 0,
        });
        close(&grammar, &mut branch.chart, 0);
        Ok(Self {
            grammar,
            branches: vec![branch],
        })
    }

    pub(crate) fn allows_token(&self, token: u32, bytes: &[u8]) -> bool {
        let mut candidate = self.clone();
        candidate.accept_token(token, bytes).is_ok()
    }

    pub(crate) fn accept_token(&mut self, token: u32, bytes: &[u8]) -> Result<(), String> {
        let mut accepted = Vec::new();
        for branch in &self.branches {
            if branch.pending_utf8.is_empty() {
                let mut token_branch = branch.clone();
                if accept_token_terminal(&self.grammar, &mut token_branch, token, bytes).is_ok()
                    && !accepted.contains(&token_branch)
                {
                    accepted.push(token_branch);
                }
            }
            if !bytes.is_empty() {
                let mut character_branch = branch.clone();
                if accept_bytes(&self.grammar, &mut character_branch, bytes).is_ok()
                    && !accepted.contains(&character_branch)
                {
                    accepted.push(character_branch);
                }
            }
        }
        if accepted.is_empty() {
            return Err(format!("token {token} is rejected by grammar"));
        }
        self.branches = accepted;
        Ok(())
    }

    pub(crate) fn allows_bytes(&self, bytes: &[u8]) -> bool {
        let mut candidate = self.clone();
        candidate.accept_bytes(bytes).is_ok()
    }

    pub(crate) fn accept_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        let mut accepted = Vec::new();
        for branch in &self.branches {
            let mut branch = branch.clone();
            if accept_bytes(&self.grammar, &mut branch, bytes).is_ok()
                && !accepted.contains(&branch)
            {
                accepted.push(branch);
            }
        }
        if accepted.is_empty() {
            return Err("bytes are rejected by grammar".into());
        }
        self.branches = accepted;
        Ok(())
    }

    pub(crate) fn is_accepting(&self) -> bool {
        self.branches.iter().any(|branch| {
            branch.pending_utf8.is_empty()
                && branch.chart.last().is_some_and(|items| {
                    items.contains(&Item {
                        production: self.grammar.start_production,
                        dot: 1,
                        origin: 0,
                    })
                })
        })
    }

    pub(crate) fn has_pending_utf8(&self) -> bool {
        self.branches
            .iter()
            .any(|branch| !branch.pending_utf8.is_empty())
    }
}

fn accept_bytes(grammar: &Grammar, branch: &mut MatchBranch, bytes: &[u8]) -> Result<(), String> {
    branch.pending_utf8.extend_from_slice(bytes);
    loop {
        match std::str::from_utf8(&branch.pending_utf8) {
            Ok(text) => {
                let characters = text.chars().map(|value| value as u32).collect::<Vec<_>>();
                branch.pending_utf8.clear();
                for character in characters {
                    accept_char(grammar, branch, character)?;
                }
                return Ok(());
            }
            Err(error) => {
                let valid = error.valid_up_to();
                if valid > 0 {
                    let prefix = std::str::from_utf8(&branch.pending_utf8[..valid])
                        .expect("UTF-8 validator supplied a valid prefix")
                        .chars()
                        .map(|value| value as u32)
                        .collect::<Vec<_>>();
                    branch.pending_utf8.drain(..valid);
                    for character in prefix {
                        accept_char(grammar, branch, character)?;
                    }
                    continue;
                }
                if error.error_len().is_some() || branch.pending_utf8.len() > 3 {
                    return Err("grammar token contains invalid UTF-8".into());
                }
                let (lower, upper) = incomplete_utf8_scalar_range(&branch.pending_utf8)
                    .ok_or_else(|| "grammar token contains invalid UTF-8".to_string())?;
                if !frontier_accepts_range(grammar, branch, lower, upper) {
                    return Err("incomplete UTF-8 prefix is rejected by grammar".into());
                }
                return Ok(());
            }
        }
    }
}

fn incomplete_utf8_scalar_range(bytes: &[u8]) -> Option<(u32, u32)> {
    let (&lead, continuations) = bytes.split_first()?;
    let (length, mut value, minimum) = match lead {
        0xC2..=0xDF => (2, u32::from(lead & 0x1F), 0x80),
        0xE0..=0xEF => (3, u32::from(lead & 0x0F), 0x800),
        0xF0..=0xF4 => (4, u32::from(lead & 0x07), 0x10000),
        _ => return None,
    };
    if bytes.len() >= length
        || continuations
            .iter()
            .any(|byte| !(0x80..=0xBF).contains(byte))
    {
        return None;
    }
    for byte in continuations {
        value = (value << 6) | u32::from(byte & 0x3F);
    }
    let remaining = length - bytes.len();
    let lower = (value << (6 * remaining)).max(minimum);
    let upper = ((value << (6 * remaining)) | ((1_u32 << (6 * remaining)) - 1)).min(0x10FFFF);

    // UTF-8 never encodes surrogate scalar values. A valid incomplete prefix can
    // only overlap the surrogate block from one side, so clipping is sufficient.
    let (lower, upper) = if lower <= 0xDFFF && upper >= 0xD800 {
        if lower < 0xD800 {
            (lower, 0xD7FF)
        } else {
            (0xE000, upper)
        }
    } else {
        (lower, upper)
    };
    (lower <= upper).then_some((lower, upper))
}

fn frontier_accepts_range(grammar: &Grammar, branch: &MatchBranch, lower: u32, upper: u32) -> bool {
    branch.chart.last().is_some_and(|items| {
        items.iter().any(|item| {
            let production = &grammar.productions[item.production];
            matches!(
                production.rhs.get(item.dot),
                Some(Symbol::Character(set)) if set.intersects(lower, upper)
            )
        })
    })
}

fn accept_char(grammar: &Grammar, branch: &mut MatchBranch, value: u32) -> Result<(), String> {
    let position = branch.chart.len() - 1;
    let mut next = HashSet::new();
    for item in &branch.chart[position] {
        let production = &grammar.productions[item.production];
        if let Some(Symbol::Character(set)) = production.rhs.get(item.dot) {
            if set.matches(value) {
                next.insert(Item {
                    production: item.production,
                    dot: item.dot + 1,
                    origin: item.origin,
                });
            }
        }
    }
    if next.is_empty() {
        return Err(format!("character U+{value:04X} is rejected by grammar"));
    }
    branch.chart.push(next);
    close(grammar, &mut branch.chart, position + 1);
    Ok(())
}

fn accept_token_terminal(
    grammar: &Grammar,
    branch: &mut MatchBranch,
    token: u32,
    bytes: &[u8],
) -> Result<(), String> {
    let position = branch.chart.len() - 1;
    let mut next = HashSet::new();
    for item in &branch.chart[position] {
        let production = &grammar.productions[item.production];
        if let Some(Symbol::Token(terminal)) = production.rhs.get(item.dot) {
            if terminal.matches(token, bytes) {
                next.insert(Item {
                    production: item.production,
                    dot: item.dot + 1,
                    origin: item.origin,
                });
            }
        }
    }
    if next.is_empty() {
        return Err(format!("token {token} does not match a token terminal"));
    }
    branch.chart.push(next);
    close(grammar, &mut branch.chart, position + 1);
    Ok(())
}

fn close(grammar: &Grammar, chart: &mut [HashSet<Item>], position: usize) {
    // Agenda-based Earley closure visits each distinct item once. The extra
    // nullable check in the predictor is what makes this a true fixed point:
    // if a zero-width child completed before a newly discovered parent was
    // queued, that parent is advanced immediately. The previous whole-set
    // rescan was correct but quadratic and made a real ATEM tool grammar spend
    // minutes in closure after every selected token.
    let mut agenda = chart[position].iter().copied().collect::<VecDeque<_>>();
    while let Some(item) = agenda.pop_front() {
        let production = &grammar.productions[item.production];
        match production.rhs.get(item.dot) {
            Some(Symbol::Rule(rule)) => {
                for &candidate in &grammar.by_lhs[*rule] {
                    let predicted = Item {
                        production: candidate,
                        dot: 0,
                        origin: position,
                    };
                    if chart[position].insert(predicted) {
                        agenda.push_back(predicted);
                    }
                }

                let nullable_complete = chart[position].iter().any(|complete| {
                    complete.origin == position
                        && grammar.productions[complete.production].lhs == *rule
                        && complete.dot == grammar.productions[complete.production].rhs.len()
                });
                if nullable_complete {
                    let advanced = Item {
                        production: item.production,
                        dot: item.dot + 1,
                        origin: item.origin,
                    };
                    if chart[position].insert(advanced) {
                        agenda.push_back(advanced);
                    }
                }
            }
            Some(Symbol::Character(_) | Symbol::Token(_)) => {}
            None => {
                let completed_rule = production.lhs;
                let parents = chart[item.origin].iter().copied().collect::<Vec<_>>();
                for parent in parents {
                    let parent_production = &grammar.productions[parent.production];
                    if matches!(
                        parent_production.rhs.get(parent.dot),
                        Some(Symbol::Rule(rule)) if *rule == completed_rule
                    ) {
                        let advanced = Item {
                            production: parent.production,
                            dot: parent.dot + 1,
                            origin: parent.origin,
                        };
                        if chart[position].insert(advanced) {
                            agenda.push_back(advanced);
                        }
                    }
                }
            }
        }
    }
}

struct Parser<'a> {
    source: &'a [u8],
    position: usize,
    names: HashMap<String, usize>,
    productions: Vec<Production>,
    generated: usize,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source: source.as_bytes(),
            position: 0,
            names: HashMap::new(),
            productions: Vec::new(),
            generated: 0,
        }
    }

    fn parse(mut self, root: &str) -> Result<Grammar, String> {
        while self.skip_layout(true) {
            let name = self.name()?;
            let lhs = self.rule_id(&name);
            self.skip_horizontal();
            self.expect(b"::=")?;
            let alternatives = self.expression(None)?;
            for rhs in alternatives {
                self.productions.push(Production { lhs, rhs });
            }
            self.consume_rule_end()?;
        }
        let root = *self
            .names
            .get(root)
            .ok_or_else(|| format!("grammar root rule {root:?} does not exist"))?;
        let synthetic = self.names.len();
        let start_production = self.productions.len();
        self.productions.push(Production {
            lhs: synthetic,
            rhs: vec![Symbol::Rule(root)],
        });
        let mut by_lhs = vec![Vec::new(); synthetic + 1];
        for (index, production) in self.productions.iter().enumerate() {
            by_lhs[production.lhs].push(index);
        }
        for (name, &rule) in &self.names {
            if by_lhs[rule].is_empty() {
                return Err(format!("grammar rule {name:?} is referenced but undefined"));
            }
        }
        Ok(Grammar {
            productions: self.productions,
            by_lhs,
            start_production,
        })
    }

    fn expression(&mut self, closing: Option<u8>) -> Result<Vec<Vec<Symbol>>, String> {
        let mut alternatives = Vec::new();
        loop {
            alternatives.push(self.sequence(closing)?);
            self.skip_horizontal();
            if self.peek() == Some(b'|') {
                self.position += 1;
                continue;
            }
            break;
        }
        Ok(alternatives)
    }

    fn sequence(&mut self, closing: Option<u8>) -> Result<Vec<Symbol>, String> {
        let mut sequence = Vec::new();
        loop {
            self.skip_horizontal();
            let Some(byte) = self.peek() else { break };
            if byte == b'|'
                || byte == b'\n'
                || byte == b'\r'
                || closing == Some(byte)
                || byte == b'#'
            {
                break;
            }
            let atom = self.atom()?;
            sequence.extend(atom);
        }
        Ok(sequence)
    }

    fn atom(&mut self) -> Result<Vec<Symbol>, String> {
        let mut body = match self.peek() {
            Some(b'"') => self.literal()?,
            Some(b'[') => vec![Symbol::Character(self.character_class()?)],
            Some(b'.') => {
                self.position += 1;
                vec![Symbol::Character(CharSet {
                    inverted: true,
                    ranges: Vec::new(),
                })]
            }
            Some(b'(') => {
                self.position += 1;
                let alternatives = self.expression(Some(b')'))?;
                self.expect(b")")?;
                vec![Symbol::Rule(self.generated_rule(alternatives))]
            }
            Some(b'<') | Some(b'!') => vec![Symbol::Token(self.token_terminal()?)],
            Some(_) => {
                let name = self.name()?;
                vec![Symbol::Rule(self.rule_id(&name))]
            }
            None => return Err("unexpected end of grammar".into()),
        };
        self.skip_horizontal();
        let repetition = match self.peek() {
            Some(b'?') => Some((0, Some(1))),
            Some(b'*') => Some((0, None)),
            Some(b'+') => Some((1, None)),
            Some(b'{') => Some(self.repetition_range()?),
            _ => None,
        };
        if repetition.is_some()
            && !matches!(self.source.get(self.position.wrapping_sub(1)), Some(b'}'))
        {
            self.position += 1;
        }
        if let Some((minimum, maximum)) = repetition {
            body = vec![Symbol::Rule(self.repeat_rule(body, minimum, maximum)?)];
        }
        Ok(body)
    }

    fn token_terminal(&mut self) -> Result<TokenTerminal, String> {
        let inverted = if self.peek() == Some(b'!') {
            self.position += 1;
            true
        } else {
            false
        };
        self.expect(b"<")?;
        let value = if self.peek() == Some(b'[') {
            self.position += 1;
            let start = self.position;
            while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                self.position += 1;
            }
            if start == self.position {
                return Err("token terminal ID must contain digits".into());
            }
            let token = std::str::from_utf8(&self.source[start..self.position])
                .expect("ASCII token ID")
                .parse::<u32>()
                .map_err(|_| "token terminal ID exceeds uint32")?;
            self.expect(b"]>")?;
            TokenValue::Id(token)
        } else {
            let start = self.position - 1;
            while self.peek().is_some_and(|byte| byte != b'>') {
                if matches!(self.peek(), Some(b'\n' | b'\r')) {
                    return Err("token terminal cannot cross a line".into());
                }
                self.position += 1;
            }
            if self.position == start + 1 {
                return Err("named token terminal must not be empty".into());
            }
            self.expect(b">")?;
            TokenValue::Piece(self.source[start..self.position].to_vec())
        };
        Ok(TokenTerminal { inverted, value })
    }

    fn repeat_rule(
        &mut self,
        body: Vec<Symbol>,
        minimum: usize,
        maximum: Option<usize>,
    ) -> Result<usize, String> {
        // Pinned llama accepts an upper bound of 5000, but rejects 5000 as
        // the required lower bound. Build bounded optionals as a linear rule
        // chain instead of materializing every repeated alternative (which
        // is quadratic in the upper bound).
        if minimum >= 5_000 || maximum.is_some_and(|maximum| minimum > maximum || maximum > 5_000) {
            return Err("invalid or excessive grammar repetition".into());
        }
        let rule = self.fresh_rule();
        match maximum {
            Some(maximum) => {
                let mut required = repeat_symbols(&body, minimum);
                let optional = maximum - minimum;
                if optional > 0 {
                    let first_tail = self.fresh_rule();
                    required.push(Symbol::Rule(first_tail));
                    let mut tail = first_tail;
                    for index in 0..optional {
                        self.productions.push(Production {
                            lhs: tail,
                            rhs: Vec::new(),
                        });
                        let mut one_more = body.clone();
                        if index + 1 < optional {
                            let next = self.fresh_rule();
                            one_more.push(Symbol::Rule(next));
                            self.productions.push(Production {
                                lhs: tail,
                                rhs: one_more,
                            });
                            tail = next;
                        } else {
                            self.productions.push(Production {
                                lhs: tail,
                                rhs: one_more,
                            });
                        }
                    }
                }
                self.productions.push(Production {
                    lhs: rule,
                    rhs: required,
                });
            }
            None => {
                let tail = self.fresh_rule();
                self.productions.push(Production {
                    lhs: tail,
                    rhs: Vec::new(),
                });
                let mut recursive = body.clone();
                recursive.push(Symbol::Rule(tail));
                self.productions.push(Production {
                    lhs: tail,
                    rhs: recursive,
                });
                let mut required = repeat_symbols(&body, minimum);
                required.push(Symbol::Rule(tail));
                self.productions.push(Production {
                    lhs: rule,
                    rhs: required,
                });
            }
        }
        Ok(rule)
    }

    fn repetition_range(&mut self) -> Result<(usize, Option<usize>), String> {
        self.position += 1;
        let minimum = self.number()?;
        let maximum = match self.peek() {
            Some(b'}') => Some(minimum),
            Some(b',') => {
                self.position += 1;
                (self.peek() != Some(b'}'))
                    .then(|| self.number())
                    .transpose()?
            }
            _ => return Err("expected ',' or '}' in repetition".into()),
        };
        self.expect(b"}")?;
        Ok((minimum, maximum))
    }

    fn literal(&mut self) -> Result<Vec<Symbol>, String> {
        self.position += 1;
        let mut output = Vec::new();
        while self.peek() != Some(b'"') {
            if self.peek().is_none() || matches!(self.peek(), Some(b'\n' | b'\r')) {
                return Err("unterminated grammar literal".into());
            }
            output.push(Symbol::Character(CharSet::one(self.codepoint()?)));
        }
        self.position += 1;
        Ok(output)
    }

    fn character_class(&mut self) -> Result<CharSet, String> {
        self.position += 1;
        let inverted = if self.peek() == Some(b'^') {
            self.position += 1;
            true
        } else {
            false
        };
        let mut ranges = Vec::new();
        while self.peek() != Some(b']') {
            if self.peek().is_none() {
                return Err("unterminated grammar character class".into());
            }
            let lower = self.codepoint()?;
            let upper =
                if self.peek() == Some(b'-') && self.source.get(self.position + 1) != Some(&b']') {
                    self.position += 1;
                    self.codepoint()?
                } else {
                    lower
                };
            if lower > upper {
                return Err("descending grammar character range".into());
            }
            ranges.push((lower, upper));
        }
        self.position += 1;
        Ok(CharSet { inverted, ranges })
    }

    fn codepoint(&mut self) -> Result<u32, String> {
        if self.peek() == Some(b'\\') {
            self.position += 1;
            let escaped = self.peek().ok_or("trailing grammar escape")?;
            self.position += 1;
            return match escaped {
                b'n' => Ok('\n' as u32),
                b'r' => Ok('\r' as u32),
                b't' => Ok('\t' as u32),
                b'"' | b'\\' | b'[' | b']' | b'-' => Ok(escaped as u32),
                b'x' => self.hex(2),
                b'u' => self.hex(4),
                b'U' => self.hex(8),
                _ => Err(format!("unsupported grammar escape \\{}", escaped as char)),
            };
        }
        let tail = std::str::from_utf8(&self.source[self.position..])
            .map_err(|_| "grammar source is not UTF-8")?;
        let character = tail.chars().next().ok_or("unexpected end of grammar")?;
        self.position += character.len_utf8();
        Ok(character as u32)
    }

    fn hex(&mut self, digits: usize) -> Result<u32, String> {
        let end = self
            .position
            .checked_add(digits)
            .ok_or("hex escape overflow")?;
        let bytes = self
            .source
            .get(self.position..end)
            .ok_or("short hex escape")?;
        let text = std::str::from_utf8(bytes).map_err(|_| "invalid hex escape")?;
        let value = u32::from_str_radix(text, 16).map_err(|_| "invalid hex escape")?;
        if char::from_u32(value).is_none() {
            return Err("hex escape is not a Unicode scalar".into());
        }
        self.position = end;
        Ok(value)
    }

    fn generated_rule(&mut self, alternatives: Vec<Vec<Symbol>>) -> usize {
        let rule = self.fresh_rule();
        for rhs in alternatives {
            self.productions.push(Production { lhs: rule, rhs });
        }
        rule
    }

    fn fresh_rule(&mut self) -> usize {
        let name = format!("__generated-{}", self.generated);
        self.generated += 1;
        self.rule_id(&name)
    }

    fn rule_id(&mut self, name: &str) -> usize {
        let next = self.names.len();
        *self.names.entry(name.to_owned()).or_insert(next)
    }

    fn name(&mut self) -> Result<String, String> {
        let start = self.position;
        while self
            .peek()
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        {
            self.position += 1;
        }
        if self.position == start {
            return Err(format!("expected grammar rule name at byte {start}"));
        }
        Ok(std::str::from_utf8(&self.source[start..self.position])
            .expect("ASCII rule name")
            .to_owned())
    }

    fn number(&mut self) -> Result<usize, String> {
        let start = self.position;
        while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.position += 1;
        }
        if start == self.position {
            return Err("expected repetition count".into());
        }
        std::str::from_utf8(&self.source[start..self.position])
            .expect("ASCII number")
            .parse()
            .map_err(|_| "invalid repetition count".into())
    }

    fn skip_layout(&mut self, newlines: bool) -> bool {
        loop {
            while self.peek().is_some_and(|byte| {
                byte == b' ' || byte == b'\t' || (newlines && matches!(byte, b'\r' | b'\n'))
            }) {
                self.position += 1;
            }
            if self.peek() == Some(b'#') {
                while self.peek().is_some_and(|byte| byte != b'\n') {
                    self.position += 1;
                }
                continue;
            }
            break;
        }
        self.position < self.source.len()
    }

    fn skip_horizontal(&mut self) {
        while self
            .peek()
            .is_some_and(|byte| byte == b' ' || byte == b'\t')
        {
            self.position += 1;
        }
    }

    fn consume_rule_end(&mut self) -> Result<(), String> {
        self.skip_horizontal();
        if self.peek() == Some(b'#') {
            while self.peek().is_some_and(|byte| byte != b'\n') {
                self.position += 1;
            }
        }
        if self.position < self.source.len() && !matches!(self.peek(), Some(b'\r' | b'\n')) {
            return Err(format!(
                "unexpected grammar input at byte {}",
                self.position
            ));
        }
        Ok(())
    }

    fn expect(&mut self, expected: &[u8]) -> Result<(), String> {
        if self
            .source
            .get(self.position..self.position + expected.len())
            != Some(expected)
        {
            return Err(format!(
                "expected {:?} at grammar byte {}",
                String::from_utf8_lossy(expected),
                self.position
            ));
        }
        self.position += expected.len();
        Ok(())
    }

    fn peek(&self) -> Option<u8> {
        self.source.get(self.position).copied()
    }
}

fn repeat_symbols(symbols: &[Symbol], count: usize) -> Vec<Symbol> {
    let mut output = Vec::with_capacity(symbols.len().saturating_mul(count));
    for _ in 0..count {
        output.extend_from_slice(symbols);
    }
    output
}

/// Deterministic JSON-schema-to-GBNF compiler for the applicable serving
/// subset. Object keys use canonical lexical order; `additionalProperties`
/// defaults to true exactly as JSON Schema specifies.
pub(crate) fn json_schema_to_gbnf(schema: &Value) -> Result<String, String> {
    let mut builder = SchemaGrammar::new(schema.clone());
    let root = builder.schema_rule(schema)?;
    builder.rules.insert("root".into(), root);
    let mut output = String::new();
    for (name, expression) in builder.rules {
        output.push_str(&name);
        output.push_str(" ::= ");
        output.push_str(&expression);
        output.push('\n');
    }
    output.push_str(JSON_RULES);
    Ok(output)
}

pub(crate) fn json_object_gbnf() -> String {
    format!("root ::= object\n{JSON_RULES}")
}

struct SchemaGrammar {
    rules: BTreeMap<String, String>,
    next: usize,
    root_schema: Value,
    refs: HashMap<String, String>,
}

impl SchemaGrammar {
    fn new(root_schema: Value) -> Self {
        Self {
            rules: BTreeMap::new(),
            next: 0,
            root_schema,
            refs: HashMap::new(),
        }
    }

    fn schema_rule(&mut self, schema: &Value) -> Result<String, String> {
        match schema {
            Value::Bool(true) => return Ok("value".into()),
            Value::Bool(false) => {
                // No Unicode scalar can satisfy this complement, so the
                // grammar is genuinely unsatisfiable rather than a literal
                // placeholder the model could accidentally emit.
                return Ok("[^\\U00000000-\\uD7FF\\uE000-\\U0010FFFF]".into());
            }
            Value::Object(_) => {}
            _ => return Err("JSON Schema must be an object or boolean".into()),
        }
        validate_schema_keywords(schema)?;
        if let Some(kind) = schema.get("type") {
            let valid = match kind {
                Value::String(kind) => is_json_schema_type(kind),
                Value::Array(kinds) => {
                    !kinds.is_empty()
                        && kinds
                            .iter()
                            .all(|kind| kind.as_str().is_some_and(is_json_schema_type))
                }
                _ => false,
            };
            if !valid {
                return Err(format!("invalid JSON Schema type {kind}"));
            }
        }
        if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
            return self.reference_rule(reference);
        }
        if let Some(value) = schema.get("const") {
            return Ok(quoted_literal(
                &serde_json::to_string(value).map_err(|e| e.to_string())?,
            ));
        }
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            if values.is_empty() {
                return Err("JSON Schema enum cannot be empty".into());
            }
            return values
                .iter()
                .map(|value| serde_json::to_string(value).map(|text| quoted_literal(&text)))
                .collect::<Result<Vec<_>, _>>()
                .map(|values| format!("({})", values.join(" | ")))
                .map_err(|error| error.to_string());
        }
        if let Some(branches) = schema
            .get("oneOf")
            .or_else(|| schema.get("anyOf"))
            .and_then(Value::as_array)
        {
            let branches = branches
                .iter()
                .map(|branch| self.schema_rule(branch))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(format!("({})", branches.join(" | ")));
        }
        if let Some(branches) = schema.get("allOf").and_then(Value::as_array) {
            return self.all_of_rule(schema, branches);
        }
        let kind = schema.get("type");
        if let Some(kinds) = kind.and_then(Value::as_array) {
            let branches = kinds
                .iter()
                .map(|kind| {
                    let mut branch = schema.clone();
                    branch["type"] = kind.clone();
                    self.schema_rule(&branch)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(format!("({})", branches.join(" | ")));
        }
        match kind.and_then(Value::as_str) {
            Some("object") => self.object_rule(schema),
            None if schema.get("properties").is_some()
                || schema.get("additionalProperties").is_some() =>
            {
                self.object_rule(schema)
            }
            Some("array") => self.array_rule(schema),
            None if schema.get("items").is_some() || schema.get("prefixItems").is_some() => {
                self.array_rule(schema)
            }
            Some("string") | None
                if schema.get("pattern").is_some()
                    || schema.get("format").is_some()
                    || schema.get("minLength").is_some()
                    || schema.get("maxLength").is_some() =>
            {
                self.string_rule(schema)
            }
            Some("string") => Ok("string".into()),
            Some("integer") => self.integer_rule(schema),
            Some("number") => Ok("number".into()),
            Some("boolean") => Ok("boolean".into()),
            Some("null") => Ok("null".into()),
            None if schema.as_object().is_some_and(serde_json::Map::is_empty) => {
                Ok("object".into())
            }
            None => Ok("value".into()),
            Some(other) => Err(format!("unsupported JSON Schema type {other:?}")),
        }
    }

    fn reference_rule(&mut self, reference: &str) -> Result<String, String> {
        if let Some(name) = self.refs.get(reference) {
            return Ok(name.clone());
        }
        let target = resolve_local_reference(&self.root_schema, reference)?;
        let name = format!("schema-ref-{}", self.next);
        self.next += 1;
        self.refs.insert(reference.into(), name.clone());
        // Install the name before descending so recursive local references
        // terminate and become ordinary recursive GBNF rules.
        self.rules.insert(name.clone(), String::new());
        let expression = self.schema_rule(&target)?;
        self.rules.insert(name.clone(), expression);
        Ok(name)
    }

    fn all_of_rule(&mut self, _parent: &Value, branches: &[Value]) -> Result<String, String> {
        if branches.is_empty() {
            return Ok("value".into());
        }
        let mut enum_intersection: Option<Vec<Value>> = None;
        let mut properties = serde_json::Map::new();
        let mut property_order = Vec::<String>::new();
        let mut required = Vec::<Value>::new();
        let mut object_components = 0usize;
        for branch in branches {
            let resolved = if let Some(reference) = branch.get("$ref").and_then(Value::as_str) {
                resolve_local_reference(&self.root_schema, reference)?
            } else {
                branch.clone()
            };
            if let Some(values) = resolved.get("enum").and_then(Value::as_array) {
                enum_intersection = Some(match enum_intersection {
                    None => values.clone(),
                    Some(current) => current
                        .into_iter()
                        .filter(|value| values.contains(value))
                        .collect(),
                });
                continue;
            }
            if let Some(component) = resolved.get("properties").and_then(Value::as_object) {
                object_components += 1;
                for (name, schema) in component {
                    if !properties.contains_key(name) {
                        property_order.push(name.clone());
                    }
                    properties.insert(name.clone(), schema.clone());
                    let name = Value::String(name.clone());
                    if !required.contains(&name) {
                        required.push(name);
                    }
                }
                continue;
            }
            if let Some(alternatives) = resolved.get("anyOf").and_then(Value::as_array) {
                object_components += 1;
                for alternative in alternatives {
                    let alternative =
                        if let Some(reference) = alternative.get("$ref").and_then(Value::as_str) {
                            resolve_local_reference(&self.root_schema, reference)?
                        } else {
                            alternative.clone()
                        };
                    let component = alternative
                        .get("properties")
                        .and_then(Value::as_object)
                        .ok_or_else(|| {
                            "allOf anyOf alternatives must resolve to object properties".to_string()
                        })?;
                    for (name, schema) in component {
                        if !properties.contains_key(name) {
                            property_order.push(name.clone());
                        }
                        properties.insert(name.clone(), schema.clone());
                    }
                }
                continue;
            }
            // String-only allOf is representable only when one component
            // supplies the effective constraint; reject combinations instead
            // of silently weakening their intersection.
            if branches.len() == 1 {
                return self.schema_rule(&resolved);
            }
            return Err("unsupported JSON Schema allOf intersection".into());
        }
        if let Some(values) = enum_intersection {
            if values.is_empty() {
                return Ok("[^\\U00000000-\\uD7FF\\uE000-\\U0010FFFF]".into());
            }
            return self.schema_rule(&serde_json::json!({ "enum": values }));
        }
        if object_components > 0 {
            let merged = serde_json::json!({
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": false,
            });
            return self.object_rule_ordered(&merged, Some(&property_order));
        }
        Err("unsupported JSON Schema allOf".into())
    }

    fn string_rule(&mut self, schema: &Value) -> Result<String, String> {
        let has_length = schema.get("minLength").is_some() || schema.get("maxLength").is_some();
        if has_length && (schema.get("pattern").is_some() || schema.get("format").is_some()) {
            return Err(
                "combined JSON Schema pattern/format and length constraints are unsupported".into(),
            );
        }
        if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
            return Ok(format!(
                "\"\\\"\" {} \"\\\"\"",
                regex_pattern_to_gbnf(pattern)?
            ));
        }
        if let Some(format) = schema.get("format").and_then(Value::as_str) {
            let rule = match format {
                "uuid" | "uuid1" | "uuid2" | "uuid3" | "uuid4" | "uuid5" => {
                    "\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\""
                }
                "date" => "\"\\\"\" date-value \"\\\"\"",
                "time" => "\"\\\"\" time-value \"\\\"\"",
                "date-time" => {
                    "\"\\\"\" date-value \"T\" time-value \"\\\"\""
                }
                // The pinned converter treats unrecognized formats (for
                // example email and URI) as annotations and falls back to an
                // ordinary JSON string.
                _ => return Ok("string".into()),
            };
            return Ok(rule.into());
        }
        let minimum = schema.get("minLength").and_then(Value::as_u64).unwrap_or(0);
        let maximum = schema.get("maxLength").and_then(Value::as_u64);
        if minimum > 2_000 || maximum.is_some_and(|value| value > 2_000 || value < minimum) {
            return Err("invalid or excessive JSON Schema string length bounds".into());
        }
        let repetition = match maximum {
            Some(maximum) if maximum == minimum => format!("{{{minimum}}}"),
            Some(maximum) => format!("{{{minimum},{maximum}}}"),
            None => format!("{{{minimum},}}"),
        };
        Ok(format!("\"\\\"\" json-character{repetition} \"\\\"\""))
    }

    fn integer_rule(&mut self, schema: &Value) -> Result<String, String> {
        for keyword in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
            if schema
                .get(keyword)
                .is_some_and(|value| value.as_i64().is_none())
            {
                return Err(format!(
                    "integer JSON Schema keyword {keyword:?} must be an int64"
                ));
            }
        }
        let minimum = schema.get("minimum").and_then(Value::as_i64).or_else(|| {
            schema
                .get("exclusiveMinimum")
                .and_then(Value::as_i64)
                .and_then(|value| value.checked_add(1))
        });
        let maximum = schema.get("maximum").and_then(Value::as_i64).or_else(|| {
            schema
                .get("exclusiveMaximum")
                .and_then(Value::as_i64)
                .and_then(|value| value.checked_sub(1))
        });
        if minimum
            .zip(maximum)
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            return Ok("[^\\U00000000-\\uD7FF\\uE000-\\U0010FFFF]".into());
        }
        if minimum.is_none() && maximum.is_none() {
            return Ok("integer".into());
        }
        Ok(format!("({})", bounded_integer_gbnf(minimum, maximum)?))
    }

    fn object_rule(&mut self, schema: &Value) -> Result<String, String> {
        self.object_rule_ordered(schema, None)
    }

    fn object_rule_ordered(
        &mut self,
        schema: &Value,
        property_order: Option<&[String]>,
    ) -> Result<String, String> {
        let properties = schema
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();
        let mut required_members = Vec::new();
        let mut optional_members = Vec::new();
        let mut property_names = Vec::new();
        let ordered = if let Some(order) = property_order {
            if order.len() != properties.len()
                || order.iter().any(|name| !properties.contains_key(name))
            {
                return Err("internal JSON Schema property order mismatch".into());
            }
            order
                .iter()
                .map(|name| (name.clone(), properties[name].clone()))
                .collect::<Vec<_>>()
        } else {
            properties.into_iter().collect::<Vec<_>>()
        };
        for (name, child) in ordered {
            let value = self.schema_rule(&child)?;
            let member = format!(
                "{} space \":\" space {value}",
                quoted_literal(&serde_json::to_string(&name).unwrap())
            );
            property_names.push(name.clone());
            if required.contains(name.as_str()) {
                required_members.push(member);
            } else {
                optional_members.push(member);
            }
        }
        if optional_members.len() > 12 {
            return Err("JSON Schema has more than 12 optional object properties".into());
        }
        let additional = schema.get("additionalProperties");
        let additional_value = match additional {
            Some(Value::Bool(true)) => Some("value".to_string()),
            Some(Value::Object(_)) => Some(self.schema_rule(additional.expect("matched"))?),
            Some(Value::Bool(false)) => None,
            None => Some("value".to_string()),
            Some(_) => return Err("additionalProperties must be a boolean or schema".into()),
        };
        let mut undeclared_required = required
            .iter()
            .copied()
            .filter(|name| !property_names.iter().any(|property| property == name))
            .collect::<Vec<_>>();
        undeclared_required.sort_unstable();
        for name in undeclared_required {
            let Some(value) = additional_value.as_deref() else {
                return Ok("[^\\U00000000-\\uD7FF\\uE000-\\U0010FFFF]".into());
            };
            required_members.push(format!(
                "{} space \":\" space {value}",
                quoted_literal(&serde_json::to_string(name).unwrap())
            ));
            property_names.push(name.to_owned());
        }
        let additional_member = additional_value
            .map(|value| unknown_object_member_rule(&property_names, &value))
            .transpose()?;

        let mut alternatives = Vec::with_capacity(1usize << optional_members.len());
        for mask in 0..(1usize << optional_members.len()) {
            let mut selected = required_members
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            for (index, member) in optional_members.iter().enumerate() {
                if mask & (1 << index) != 0 {
                    selected.push(member);
                }
            }
            let fixed = selected.join(" comma ");
            let body = match (&additional_member, fixed.is_empty()) {
                (Some(member), true) => format!("{member} (comma {member})*"),
                (Some(member), false) => format!("{fixed} (comma {member})*"),
                (None, _) => fixed,
            };
            alternatives.push(format!("\"{{\" space {body} space \"}}\""));
        }
        Ok(format!("({})", alternatives.join(" | ")))
    }

    fn array_rule(&mut self, schema: &Value) -> Result<String, String> {
        // Preserve the source-pinned converter's field precedence and legacy
        // spelling. `items` wins when both fields exist, a schema-valued
        // `prefixItems` is homogeneous, and an array value is a fixed tuple.
        let selected = schema.get("items").or_else(|| schema.get("prefixItems"));
        if let Some(tuple) = selected.and_then(Value::as_array) {
            if tuple.is_empty() {
                return Ok("\"[\" space space \"]\"".into());
            }
            let prefix_rules = tuple
                .iter()
                .map(|item| self.schema_rule(item))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(format!(
                "\"[\" space {} space \"]\"",
                array_items_body(&prefix_rules)
            ));
        }
        let item = self.schema_rule(selected.unwrap_or(&Value::Bool(true)))?;
        let minimum = schema.get("minItems").and_then(Value::as_u64).unwrap_or(0);
        let maximum = schema.get("maxItems").and_then(Value::as_u64);
        if minimum > 2_000 || maximum.is_some_and(|value| value > 2_000 || value < minimum) {
            return Err("invalid or excessive JSON Schema array bounds".into());
        }
        let item = format!("({item})");
        let contents = match maximum {
            Some(maximum) => {
                let alternatives = (minimum..=maximum)
                    .map(|count| {
                        (0..count)
                            .map(|index| {
                                if index == 0 {
                                    item.clone()
                                } else {
                                    format!("comma {item}")
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .collect::<Vec<_>>();
                format!("({})", alternatives.join(" | "))
            }
            None if minimum == 0 => format!("({item} (comma {item})*)?"),
            None => {
                let required = (0..minimum)
                    .map(|index| {
                        if index == 0 {
                            item.clone()
                        } else {
                            format!("comma {item}")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("{required} (comma {item})*")
            }
        };
        Ok(format!("\"[\" space {contents} space \"]\""))
    }
}

fn resolve_local_reference(root: &Value, reference: &str) -> Result<Value, String> {
    if !reference.starts_with("#/") {
        return Err(format!(
            "only local JSON Schema references are supported, got {reference:?}"
        ));
    }
    let mut target = root;
    for raw in reference[2..].split('/') {
        let token = raw.replace("~1", "/").replace("~0", "~");
        target = match target {
            Value::Object(object) => object.get(&token),
            Value::Array(array) => token
                .parse::<usize>()
                .ok()
                .and_then(|index| array.get(index)),
            _ => None,
        }
        .ok_or_else(|| format!("unresolved JSON Schema reference {reference:?}"))?;
    }
    Ok(target.clone())
}

fn validate_schema_keywords(schema: &Value) -> Result<(), String> {
    let Some(object) = schema.as_object() else {
        return Ok(());
    };
    // Annotation and definition-container keywords do not change the
    // accepted language. Every assertion keyword not represented by the
    // compiler is rejected explicitly so structured output can never be
    // weaker than the client-requested schema.
    const SUPPORTED: &[&str] = &[
        "$schema",
        "$id",
        "$anchor",
        "$comment",
        "$ref",
        "$defs",
        "definitions",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "const",
        "enum",
        "oneOf",
        "anyOf",
        "allOf",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "pattern",
        "format",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "prefixItems",
        "items",
        "minItems",
        "maxItems",
    ];
    if let Some(keyword) = object.keys().find(|key| !SUPPORTED.contains(&key.as_str())) {
        return Err(format!(
            "unsupported JSON Schema assertion keyword {keyword:?}"
        ));
    }
    if schema.get("type").and_then(Value::as_str) == Some("number")
        && ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
            .iter()
            .any(|keyword| schema.get(*keyword).is_some())
    {
        return Err("bounded JSON Schema numbers are unsupported; use integer bounds".into());
    }
    if let Some(required) = schema.get("required") {
        let values = required
            .as_array()
            .ok_or_else(|| "JSON Schema required must be an array".to_string())?;
        if values.iter().any(|value| value.as_str().is_none()) {
            return Err("JSON Schema required entries must be strings".into());
        }
    }
    Ok(())
}

fn is_json_schema_type(kind: &str) -> bool {
    matches!(
        kind,
        "object" | "array" | "string" | "integer" | "number" | "boolean" | "null"
    )
}

pub(crate) fn quoted_literal(value: &str) -> String {
    let mut output = String::from("\"");
    for character in value.chars() {
        match character {
            '\\' => output.push_str("\\\\"),
            '"' => output.push_str("\\\""),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

fn array_items_body(items: &[String]) -> String {
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            if index == 0 {
                item.clone()
            } else {
                format!("comma {item}")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Default)]
struct ExcludedKeyTrie {
    children: BTreeMap<char, ExcludedKeyTrie>,
    terminal: bool,
}

fn unknown_object_member_rule(names: &[String], value_rule: &str) -> Result<String, String> {
    let mut trie = ExcludedKeyTrie::default();
    for name in names {
        let encoded = serde_json::to_string(name).map_err(|error| error.to_string())?;
        let encoded = encoded
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .ok_or_else(|| "JSON property name did not serialize as a string".to_string())?;
        let mut node = &mut trie;
        for character in encoded.chars() {
            node = node.children.entry(character).or_default();
        }
        node.terminal = true;
    }
    let key = if names.is_empty() {
        "string".into()
    } else {
        format!("\"\\\"\" {} \"\\\"\"", excluded_key_body(&trie)?)
    };
    Ok(format!("{key} \":\" space {value_rule}"))
}

fn excluded_key_body(node: &ExcludedKeyTrie) -> Result<String, String> {
    let mut alternatives = Vec::new();
    for (character, child) in &node.children {
        let suffix = if child.children.is_empty() {
            "json-character+".into()
        } else {
            excluded_key_body(child)?
        };
        alternatives.push(format!(
            "{} {suffix}",
            quoted_literal(&character.to_string())
        ));
    }
    if !node.children.is_empty() {
        let rejected = node
            .children
            .keys()
            .map(|character| match character {
                '\\' | ']' | '-' | '^' => format!("\\{character}"),
                other => other.to_string(),
            })
            .collect::<String>();
        alternatives.push(format!("[^\"{rejected}] json-character*"));
    }
    if alternatives.is_empty() {
        return Ok("json-character+".into());
    }
    let body = format!("({})", alternatives.join(" | "));
    Ok(if node.terminal {
        body
    } else {
        format!("{body}?")
    })
}

fn bounded_integer_gbnf(minimum: Option<i64>, maximum: Option<i64>) -> Result<String, String> {
    let mut alternatives = Vec::new();
    let negative_allowed = minimum.is_none_or(|value| value < 0);
    if negative_allowed {
        let smallest_abs = match maximum {
            Some(maximum) if maximum < 0 => maximum.unsigned_abs(),
            _ => 0,
        };
        let largest_abs = minimum.filter(|value| *value < 0).map(i64::unsigned_abs);
        let magnitude = decimal_range_gbnf(Some(smallest_abs), largest_abs)?;
        alternatives.push(format!("\"-\" ({magnitude})"));
    }
    if maximum.is_none_or(|value| value >= 0) {
        let lower = minimum.map_or(0, |value| value.max(0) as u64);
        let upper = maximum
            .filter(|value| *value >= 0)
            .map(|value| value as u64);
        alternatives.push(decimal_range_gbnf(Some(lower), upper)?);
    }
    if alternatives.is_empty() {
        return Err("integer range has no values".into());
    }
    Ok(alternatives.join(" | "))
}

fn decimal_range_gbnf(minimum: Option<u64>, maximum: Option<u64>) -> Result<String, String> {
    const MAX_PINNED_INTEGER: u64 = 9_999_999_999_999_999;
    let minimum = minimum.unwrap_or(0);
    if maximum.is_some_and(|maximum| maximum < minimum) {
        return Err("descending integer range".into());
    }
    if minimum > MAX_PINNED_INTEGER {
        return Err("integer bound exceeds pinned 16-digit grammar".into());
    }
    let maximum = maximum.map(|value| value.min(MAX_PINNED_INTEGER));
    let minimum_digits = minimum.to_string().len();
    let maximum_digits = maximum.map_or(16, |value| value.to_string().len());
    let mut alternatives = Vec::new();
    for digits in minimum_digits..=maximum_digits {
        let floor = if digits == 1 {
            0
        } else {
            10u64
                .checked_pow((digits - 1) as u32)
                .ok_or_else(|| "integer bound is too wide".to_string())?
        };
        let ceiling = 10u64
            .checked_pow(digits as u32)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| "integer bound is too wide".to_string())?;
        let lower = minimum.max(floor);
        let upper = maximum.unwrap_or(ceiling).min(ceiling);
        if lower <= upper {
            alternatives.push(same_width_decimal_range(
                &lower.to_string(),
                &upper.to_string(),
            )?);
        }
    }
    if alternatives.is_empty() {
        return Err("integer range has no pinned 16-digit values".into());
    }
    Ok(alternatives.join(" | "))
}

fn same_width_decimal_range(lower: &str, upper: &str) -> Result<String, String> {
    if lower.len() != upper.len() || lower > upper {
        return Err("decimal range endpoints differ in width or order".into());
    }
    if lower == upper {
        return Ok(quoted_literal(lower));
    }
    if lower.bytes().all(|digit| digit == b'0') && upper.bytes().all(|digit| digit == b'9') {
        return Ok(match lower.len() {
            1 => "[0-9]".into(),
            count => format!("[0-9]{{{count}}}"),
        });
    }
    if lower.starts_with('1')
        && lower[1..].bytes().all(|digit| digit == b'0')
        && upper.bytes().all(|digit| digit == b'9')
    {
        return Ok(match lower.len() {
            1 => "[1-9]".into(),
            2 => "[1-9] [0-9]".into(),
            count => format!("[1-9] [0-9]{{{}}}", count - 1),
        });
    }
    let common = lower
        .bytes()
        .zip(upper.bytes())
        .take_while(|(left, right)| left == right)
        .count();
    let prefix = &lower[..common];
    let lower_tail = &lower[common..];
    let upper_tail = &upper[common..];
    let lower_digit = lower_tail.as_bytes()[0];
    let upper_digit = upper_tail.as_bytes()[0];
    let remaining = lower_tail.len() - 1;
    let suffix = |count: usize| {
        if count == 0 {
            String::new()
        } else if count == 1 {
            " [0-9]".into()
        } else {
            format!(" [0-9]{{{count}}}")
        }
    };
    let mut choices = Vec::new();
    let low_suffix = &lower_tail[1..];
    let high_suffix = &upper_tail[1..];
    choices.push(format!(
        "{} {}",
        quoted_literal(&(lower_digit as char).to_string()),
        if remaining == 0 {
            String::new()
        } else {
            same_width_decimal_range(low_suffix, &"9".repeat(remaining))?
        }
    ));
    if lower_digit + 1 < upper_digit {
        choices.push(format!(
            "[{}-{}]{}",
            (lower_digit + 1) as char,
            (upper_digit - 1) as char,
            suffix(remaining)
        ));
    }
    choices.push(format!(
        "{} {}",
        quoted_literal(&(upper_digit as char).to_string()),
        if remaining == 0 {
            String::new()
        } else {
            same_width_decimal_range(&"0".repeat(remaining), high_suffix)?
        }
    ));
    let body = format!("({})", choices.join(" | "));
    Ok(if prefix.is_empty() {
        body
    } else {
        format!("{} {body}", quoted_literal(prefix))
    })
}

fn regex_pattern_to_gbnf(pattern: &str) -> Result<String, String> {
    let anchored_start = pattern.starts_with('^');
    let anchored_end = pattern.ends_with('$') && !pattern.ends_with("\\$");
    let body = &pattern
        [usize::from(anchored_start)..pattern.len().saturating_sub(usize::from(anchored_end))];
    let mut output = Vec::<String>::new();
    let mut chars = body.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '\\' => {
                let escaped = chars
                    .next()
                    .ok_or_else(|| "pattern ends in backslash".to_string())?;
                match escaped {
                    'd' => output.push("[0-9]".into()),
                    'D' => output.push("[^0-9]".into()),
                    's' => output.push("[ \\t\\n\\r]".into()),
                    'S' => output.push("[^ \\t\\n\\r]".into()),
                    'w' => output.push("[A-Za-z0-9_]".into()),
                    'W' => output.push("[^A-Za-z0-9_]".into()),
                    'n' => output.push(quoted_literal("\n")),
                    'r' => output.push(quoted_literal("\r")),
                    't' => output.push(quoted_literal("\t")),
                    other => output.push(quoted_literal(&other.to_string())),
                }
            }
            '[' => {
                let mut class = String::from("[");
                let mut closed = false;
                let mut escaped = false;
                for next in chars.by_ref() {
                    class.push(next);
                    if next == ']' && !escaped {
                        closed = true;
                        break;
                    }
                    escaped = next == '\\' && !escaped;
                    if next != '\\' {
                        escaped = false;
                    }
                }
                if !closed {
                    return Err("unterminated JSON Schema pattern character class".into());
                }
                output.push(class);
            }
            '(' => {
                if chars.peek() == Some(&'?') {
                    chars.next();
                    if chars.next() != Some(':') {
                        return Err("only noncapturing (?:...) pattern groups are supported".into());
                    }
                }
                output.push("(".into());
            }
            '{' => {
                let mut repetition = String::from("{");
                let mut closed = false;
                for next in chars.by_ref() {
                    repetition.push(next);
                    if next == '}' {
                        closed = true;
                        break;
                    }
                    if !next.is_ascii_digit() && next != ',' {
                        return Err("invalid JSON Schema pattern repetition".into());
                    }
                }
                if !closed {
                    return Err("unterminated JSON Schema pattern repetition".into());
                }
                let bounds = &repetition[1..repetition.len() - 1];
                let valid = if let Some((minimum, maximum)) = bounds.split_once(',') {
                    !minimum.is_empty()
                        && minimum.chars().all(|character| character.is_ascii_digit())
                        && maximum.chars().all(|character| character.is_ascii_digit())
                } else {
                    !bounds.is_empty() && bounds.chars().all(|character| character.is_ascii_digit())
                };
                if !valid || output.is_empty() {
                    return Err("invalid JSON Schema pattern repetition".into());
                }
                output.push(repetition);
            }
            ')' | '|' | '?' | '*' | '+' => output.push(character.to_string()),
            '.' => {
                output.push("regex-dot".into());
            }
            '^' | '$' => output.push(quoted_literal(&character.to_string())),
            other => output.push(quoted_literal(&other.to_string())),
        }
    }
    let expression = format!("({})", output.join(" "));
    Ok(match (anchored_start, anchored_end) {
        (true, true) => expression,
        (true, false) => format!("({expression} json-character*)"),
        (false, true) => format!("(json-character* {expression})"),
        (false, false) => format!("(json-character* {expression} json-character*)"),
    })
}

const JSON_RULES: &str = r#"
value ::= object | array | string | number | boolean | null
object ::= "{" space (string space ":" space value (comma string space ":" space value)*)? space "}"
array ::= "[" space (value (comma value)*)? space "]"
string ::= "\"" characters "\""
characters ::= json-character*
json-character ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
regex-dot ::= [^\x0a\x0d]
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
integer ::= "-"? ("0" | [1-9] [0-9]*)
boolean ::= "true" | "false"
null ::= "null"
comma ::= "," space
space ::= [ ]{0,1} | "\n"{1,2} [ \t]{0,20}
ws ::= [ \t\n\r]*
date-value ::= [0-9]{4} "-" ("0" [1-9] | "1" [0-2]) "-" ("0" [1-9] | [12] [0-9] | "3" [01])
time-value ::= ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ("." [0-9]{3})? ("Z" | [+-] ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9])
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambiguous_gbnf_and_utf8_fragments_match_incrementally() {
        let mut grammar =
            GrammarMatcher::parse("root ::= (\"ab\" | \"a\" [β-δ]+) \"!\"?\n", "root").unwrap();
        assert!(grammar.allows_bytes(b"a"));
        grammar.accept_bytes(b"a").unwrap();
        let beta = "β".as_bytes();
        assert!(grammar.allows_bytes(&beta[..1]));
        grammar.accept_bytes(&beta[..1]).unwrap();
        assert!(!grammar.is_accepting());
        grammar.accept_bytes(&beta[1..]).unwrap();
        assert!(grammar.is_accepting());
        assert!(!grammar.allows_bytes("z".as_bytes()));
        grammar.accept_bytes(b"!").unwrap();
        assert!(grammar.is_accepting());
    }

    #[test]
    fn repetition_and_inverse_classes_follow_gbnf() {
        let mut grammar = GrammarMatcher::parse("root ::= [^0-9]{2,3}\n", "root").unwrap();
        grammar.accept_bytes(b"ab").unwrap();
        assert!(grammar.is_accepting());
        assert!(grammar.allows_bytes(b"c"));
        assert!(!grammar.allows_bytes(b"4"));

        assert!(GrammarMatcher::parse("root ::= \"a\"{0,5000}\n", "root").is_ok());
        assert!(GrammarMatcher::parse("root ::= \"a\"{3,5000}\n", "root").is_ok());
        assert!(GrammarMatcher::parse("root ::= \"a\"{5000}\n", "root").is_err());
        assert!(GrammarMatcher::parse("root ::= \"a\"{5000,}\n", "root").is_err());
    }

    #[test]
    fn token_terminals_coexist_with_character_paths() {
        let source = "root ::= (<[42]> | \"x\" | <|eom|>) !<[99]>\n";

        let mut by_id = GrammarMatcher::parse(source, "root").unwrap();
        by_id.accept_token(42, b"not-the-piece").unwrap();
        by_id.accept_token(7, b"z").unwrap();
        assert!(by_id.is_accepting());

        let mut by_character = GrammarMatcher::parse(source, "root").unwrap();
        by_character.accept_token(1, b"x").unwrap();
        assert!(by_character.accept_token(99, b"other").is_err());

        let mut by_name = GrammarMatcher::parse(source, "root").unwrap();
        by_name.accept_token(200_008, b"<|eom|>").unwrap();
        by_name.accept_token(1, b"!").unwrap();
        assert!(by_name.is_accepting());

        let mut bytes_only = GrammarMatcher::parse("root ::= <[42]>\n", "root").unwrap();
        assert!(bytes_only.accept_bytes(b"anything").is_err());
        let mut empty_piece = GrammarMatcher::parse("root ::= <[42]>\n", "root").unwrap();
        empty_piece.accept_token(42, b"").unwrap();
        assert!(empty_piece.is_accepting());
    }

    #[test]
    fn leading_space_literal_accepts_the_single_token_piece() {
        let mut grammar = GrammarMatcher::parse(r#"root ::= " yes" | " no""#, "root").unwrap();
        assert!(grammar.allows_token(19_690, b" yes"));
        assert!(grammar.allows_token(916, b" no"));
        grammar.accept_token(220, b" ").unwrap();
        assert!(!grammar.allows_token(5_677, &[233, 166]));
    }

    #[test]
    fn schema_compiler_produces_a_matching_json_grammar() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"enum": ["yes", "no"]}},
            "required": ["answer"],
            "additionalProperties": false
        });
        let source = json_schema_to_gbnf(&schema).unwrap();
        let mut grammar = GrammarMatcher::parse(&source, "root").unwrap();
        grammar.accept_bytes(br#"{"answer":"yes"}"#).unwrap();
        assert!(grammar.is_accepting());
    }

    #[test]
    fn source_space_rule_is_nullable_and_bounded() {
        let source = "root ::= \"[\" space \"a\" space \"]\"\nspace ::= [ ]{0,1} | \"\\n\"{1,2} [ \\t]{0,20}\n";
        let accepts = |text: &str| {
            let mut matcher = GrammarMatcher::parse(source, "root").unwrap();
            matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_accepting()
        };
        assert!(accepts("[a]"));
        assert!(accepts("[ a ]"));
        assert!(!accepts("[  a]"));
        assert!(!accepts("[\r\na]"));
    }

    #[test]
    fn object_schema_preserves_declared_constraints_and_additional_policy() {
        let open_by_default = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"const": "yes"}},
            "required": ["answer"]
        });
        let grammar = json_schema_to_gbnf(&open_by_default).unwrap();
        let accepts = |text: &str| {
            let mut matcher = GrammarMatcher::parse(&grammar, "root").unwrap();
            matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_accepting()
        };
        assert!(accepts(r#"{"answer":"yes"}"#));
        assert!(!accepts(r#"{"answer":"no"}"#));
        assert!(accepts(r#"{"answer":"yes","extra":1}"#));

        let closed = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"const": "yes"}},
            "required": ["answer"],
            "additionalProperties": false
        });
        let grammar = json_schema_to_gbnf(&closed).unwrap();
        let accepts = |text: &str| {
            let mut matcher = GrammarMatcher::parse(&grammar, "root").unwrap();
            matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_accepting()
        };
        assert!(!accepts(r#"{"answer":"yes","extra":1}"#));
        assert!(!accepts(r#"{"answer":"yes","answer":1}"#));

        let escaped = serde_json::json!({
            "type": "object",
            "properties": {"quoted\"key": {"type": "integer"}},
            "additionalProperties": true
        });
        let grammar = json_schema_to_gbnf(&escaped).unwrap();
        let accepts = |text: &str| {
            let mut matcher = GrammarMatcher::parse(&grammar, "root").unwrap();
            matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_accepting()
        };
        assert!(accepts(r#"{"quoted\"key":1}"#));
        assert!(accepts(r#"{"other":1}"#));
        assert!(!accepts(r#"{"quoted\"key":"wrong"}"#));
    }

    #[test]
    fn pinned_array_field_precedence_and_tuple_semantics() {
        let schema = serde_json::json!({
            "type":"array",
            "prefixItems":[{"type":"string"}, {"type":"integer"}],
            "items":{"type":"boolean"},
            "minItems":1,
            "maxItems":4
        });
        let source = json_schema_to_gbnf(&schema).unwrap();
        let accepts = |text: &str| {
            let mut matcher = GrammarMatcher::parse(&source, "root").unwrap();
            matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_accepting()
        };
        // Source-pinned llama prefers items and therefore ignores the
        // simultaneous prefix tuple.
        assert!(accepts("[true]"));
        assert!(accepts("[true,false]"));
        assert!(!accepts("[]"));
        assert!(!accepts(r#"["x",2]"#));
        assert!(!accepts("[true,false,true,false,true]"));

        let fixed = serde_json::json!({
            "prefixItems":[{"type":"string"}, {"type":"integer"}]
        });
        let source = json_schema_to_gbnf(&fixed).unwrap();
        let accepts = |text: &str| {
            let mut matcher = GrammarMatcher::parse(&source, "root").unwrap();
            matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_accepting()
        };
        assert!(accepts(r#"["x",2]"#));
        assert!(!accepts(r#"["x"]"#));

        let homogeneous = serde_json::json!({
            "type":"array",
            "prefixItems":{"type":"string"}
        });
        let source = json_schema_to_gbnf(&homogeneous).unwrap();
        let accepts = |text: &str| {
            let mut matcher = GrammarMatcher::parse(&source, "root").unwrap();
            matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_accepting()
        };
        assert!(accepts("[]"), "{source}");
        assert!(accepts(r#"["x","y"]"#));
        assert!(!accepts(r#"["x",2]"#));
    }

    #[test]
    fn schema_constraints_are_never_silently_weakened() {
        for schema in [
            serde_json::json!({"type":"array", "uniqueItems":true}),
            serde_json::json!({"type":"number", "minimum":0.5}),
            serde_json::json!({"type":"string", "pattern":"^[a-z]+$", "minLength":2}),
        ] {
            assert!(json_schema_to_gbnf(&schema).is_err(), "{schema}");
        }

        let open = serde_json::json!({
            "type":"object",
            "required":["witness"]
        });
        let source = json_schema_to_gbnf(&open).unwrap();
        let mut matcher = GrammarMatcher::parse(&source, "root").unwrap();
        matcher.accept_bytes(br#"{"witness":1}"#).unwrap();
        assert!(matcher.is_accepting());

        let impossible = serde_json::json!({
            "type":"object",
            "required":["witness"],
            "additionalProperties":false
        });
        let source = json_schema_to_gbnf(&impossible).unwrap();
        let mut matcher = GrammarMatcher::parse(&source, "root").unwrap();
        assert!(matcher.accept_bytes(br#"{"witness":1}"#).is_err());
    }

    #[test]
    fn source_pinned_json_schema_fixture_families_compile() {
        // One schema from every distinct family in the pinned upstream
        // converter corpus. Repetitions of the integer-bound family are
        // exercised exhaustively below.
        let schemas = [
            r##"{}"##,
            r##"{"items":[{"format":"date"},{"format":"uuid"},{"format":"time"},{"format":"date-time"}]}"##,
            r##"{"type":"string","minLength":1,"maxLength":4}"##,
            r##"{"enum":["red","amber","green",null,42,["foo"]]}"##,
            r##"{"type":["array","null"],"prefixItems":{"type":"string"}}"##,
            r##"{"prefixItems":[{"type":"string"},{"type":"number"}]}"##,
            r##"{"type":"array","items":{},"prefixItems":{"type":"string"}}"##,
            r##"{"items":{"type":["number","integer"]},"minItems":3,"maxItems":5}"##,
            r##"{"type":"string","pattern":"^abc?d*efg+(hij)?kl$"}"##,
            r##"{"type":"string","pattern":"^\\[\\]\\{\\}\\(\\)\\|\\+\\*\\?$"}"##,
            r##"{"type":"string","pattern":"^A|B|C|D$"}"##,
            r##"{"type":"string","pattern":"^(\\([0-9]{1,3}\\))?[0-9]{3}-[0-9]{4} a{3,5}nd...$"}"##,
            r##"{"type":"object","properties":{"b":{"type":"string"},"c":{"type":"string"},"a":{"type":"string"}},"required":["a","b","c"],"additionalProperties":false,"definitions":{}}"##,
            r##"{"properties":{"and":{"type":"number"},"also":{"type":"number"}},"required":["and"],"additionalProperties":{"type":"number"}}"##,
            r##"{"properties":{"":{"type":"integer"},"a":{"type":"integer"}},"additionalProperties":{"type":"integer"}}"##,
            r##"{"$ref":"#/definitions/foo","definitions":{"foo":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}}}"##,
            r##"{"properties":{"a":{"anyOf":[{"type":"string"},{"type":"number"}]},"b":{"anyOf":[{"$ref":"#/properties/a/anyOf/0"},{"type":"boolean"}]}},"type":"object"}"##,
            r##"{"allOf":[{"$ref":"#/definitions/foo"},{"$ref":"#/definitions/bar"},{"anyOf":[{"$ref":"#/definitions/baz"},{"$ref":"#/definitions/bam"}]}],"definitions":{"foo":{"properties":{"a":{"type":"number"}}},"bar":{"properties":{"b":{"type":"number"}}},"bam":{"properties":{"c":{"type":"number"}}},"baz":{"properties":{"d":{"type":"number"}}}},"type":"object"}"##,
            r##"{"allOf":[{"$ref":"#/definitions/foo"},{"$ref":"#/definitions/bar"}],"definitions":{"foo":{"type":"string","enum":["a","b","c"]},"bar":{"type":"string","enum":["b","c","d"]}}}"##,
            r##"{"description":"annotation-only schemas remain unconstrained"}"##,
            r##"{"properties":{"code":{"const":" \r \n \" \\ ","description":"Generated code","title":"Code","type":"string"}},"required":["code"],"title":"DecoderResponse","type":"object"}"##,
        ];
        for source in schemas {
            let schema: Value = serde_json::from_str(source).unwrap();
            let grammar = json_schema_to_gbnf(&schema)
                .unwrap_or_else(|error| panic!("schema failed: {source}: {error}"));
            GrammarMatcher::parse(&grammar, "root")
                .unwrap_or_else(|error| panic!("grammar failed: {source}: {error}\n{grammar}"));
        }

        for minimum in [-123, -10, -5, 0, 1, 3, 9, 10, 15, 25] {
            let schema = serde_json::json!({"type":"integer","minimum":minimum});
            let grammar = json_schema_to_gbnf(&schema).unwrap();
            GrammarMatcher::parse(&grammar, "root").unwrap();
        }
        for maximum in [-5, 1, 23, 30, 42, 100, 300] {
            let schema = serde_json::json!({"type":"integer","maximum":maximum});
            let grammar = json_schema_to_gbnf(&schema).unwrap();
            GrammarMatcher::parse(&grammar, "root").unwrap();
        }
    }

    #[test]
    fn pinned_integer_regex_and_all_of_semantics_match_values() {
        let accepts = |schema: Value, text: &str| {
            let grammar = json_schema_to_gbnf(&schema).unwrap();
            let mut matcher = GrammarMatcher::parse(&grammar, "root").unwrap();
            matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_accepting()
        };

        let minimum = serde_json::json!({"type":"integer","minimum":-123});
        for value in ["-123", "-0", "0", "9999999999999999"] {
            assert!(accepts(minimum.clone(), value), "{value}");
        }
        for value in ["-124", "10000000000000000"] {
            assert!(!accepts(minimum.clone(), value), "{value}");
        }

        let maximum = serde_json::json!({"type":"integer","maximum":-5});
        assert!(accepts(maximum.clone(), "-5"));
        assert!(accepts(maximum.clone(), "-999"));
        assert!(!accepts(maximum.clone(), "-4"));
        assert!(!accepts(maximum, "0"));

        let range = serde_json::json!({"type":"integer","minimum":15,"maximum":300});
        for value in ["15", "99", "100", "300"] {
            assert!(accepts(range.clone(), value), "{value}");
        }
        for value in ["14", "301", "-15"] {
            assert!(!accepts(range.clone(), value), "{value}");
        }

        let pattern = serde_json::json!({
            "type":"string",
            "pattern":"^(\\([0-9]{1,3}\\))?[0-9]{3}-[0-9]{4} a{3,5}nd...$"
        });
        assert!(accepts(pattern.clone(), r#""(12)345-6789 aaand...""#));
        assert!(!accepts(pattern, r#""(1234)345-6789 aand...""#));

        let all_of = serde_json::json!({
            "allOf":[
                {"$ref":"#/definitions/foo"},
                {"$ref":"#/definitions/bar"},
                {"anyOf":[
                    {"$ref":"#/definitions/baz"},
                    {"$ref":"#/definitions/bam"}
                ]}
            ],
            "definitions":{
                "foo":{"properties":{"a":{"type":"number"}}},
                "bar":{"properties":{"b":{"type":"number"}}},
                "bam":{"properties":{"c":{"type":"number"}}},
                "baz":{"properties":{"d":{"type":"number"}}}
            },
            "type":"object"
        });
        assert!(accepts(all_of.clone(), r#"{"a":1,"b":2}"#));
        assert!(accepts(all_of.clone(), r#"{"a":1,"b":2,"d":3,"c":4}"#));
        assert!(!accepts(all_of.clone(), r#"{"a":1}"#));
        assert!(!accepts(all_of, r#"{"a":1,"b":2,"extra":3}"#));
    }
}
