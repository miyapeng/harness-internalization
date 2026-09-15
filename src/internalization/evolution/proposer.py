from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict
from ..core.types import Cost, write_json

class APITransport:
    def __init__(self, *, model=None, base_url=None, api_key=None, transport=None, max_tokens=8192, temperature=0.7):
        self.model, self.base_url, self.api_key = model, base_url, api_key
        self.transport = transport
        self.max_tokens,self.temperature=max_tokens,temperature

    def request_json(self,contract,public,output):
        """Shared transport for code proposals; endpoint selection remains protected configuration."""
        payload={"model":self.model or os.environ["HI_PROPOSER_MODEL"],
            "messages":[{"role":"system","content":contract},{"role":"user","content":json.dumps(public,ensure_ascii=False)}],
            "max_tokens":self.max_tokens,"temperature":self.temperature,"response_format":{"type":"json_object"}}
        output.mkdir(parents=True,exist_ok=True)
        write_json(output/"proposer_prompt.json",payload)
        start=time.monotonic()
        if self.transport is not None: raw=self.transport(payload)
        else:
            url=(self.base_url or os.environ["HI_PROPOSER_BASE_URL"]).rstrip("/")+"/chat/completions"
            http=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={
                "Authorization":"Bearer "+(self.api_key or os.environ["HI_PROPOSER_API_KEY"]),"Content-Type":"application/json"})
            with urllib.request.urlopen(http,timeout=300) as response: raw=json.load(response)
        write_json(output/"proposer_response.json",raw)
        self.last_cost=Cost(raw["usage"]["prompt_tokens"],raw["usage"]["completion_tokens"],1,1,latency_s=time.monotonic()-start)
        write_json(output/"cost.json",asdict(self.last_cost))
        return json.loads(raw["choices"][0]["message"]["content"])
