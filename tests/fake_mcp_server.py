"""测试用的假 MCP 服务器：按工具名模拟正常返回、isError、超长单行、卡住和中途退出。"""
import json, sys, time

for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    mid, method = msg["id"], msg["method"]
    if method == "initialize":
        res = {"protocolVersion": "2024-11-05", "capabilities": {}}
    elif method == "tools/list":
        res = {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}
    else:
        name = msg["params"]["name"]
        if name == "big":
            res = {"content": [{"type": "text", "text": "x" * (20 * 1024 * 1024)}]}
        elif name == "hang":
            time.sleep(30)
            continue
        elif name == "fail":
            res = {"content": [{"type": "text", "text": "tool failed"}], "isError": True}
        elif name == "die":
            print("fatal: something broke in the server", file=sys.stderr, flush=True)
            sys.exit(1)
        else:
            res = {"content": [{"type": "text", "text": "echo:" + json.dumps(msg["params"]["arguments"])}]}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": res}) + "\n")
    sys.stdout.flush()
