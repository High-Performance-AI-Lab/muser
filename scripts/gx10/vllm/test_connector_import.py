#!/usr/bin/env python3
"""Fail-fast packaging check for the resident Handoff V2 connector."""

from __future__ import annotations

import json

import muser_v2_send
from muser_vllm import connector


print(
    json.dumps(
        {
            "schema": "muser.spark-connector-import.v1",
            "connector": connector.MuserMuseHandoffConnector.__name__,
            "sender": muser_v2_send.DeferredHandoffV2Sender.__name__,
            "protocol": muser_v2_send.PROTOCOL,
            "status": "pass",
        },
        sort_keys=True,
    )
)
