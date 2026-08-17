import os
import time
import requests
import psutil

URL = "http://127.0.0.1:8000/api/v1/ingestion/upload"
PDF = "data/sample.pdf"

TOTAL_REQUESTS = 5

process = psutil.Process(os.getpid())

print("========== PDF LOAD + MEMORY TEST ==========")
print("PDF:", PDF)
print("Requests:", TOTAL_REQUESTS)

if not os.path.exists(PDF):
    print("PDF NOT FOUND")
    raise SystemExit(1)

# Find the Python process listening on port 8000
server_process = None

for conn in psutil.net_connections(kind="inet"):
    if conn.laddr and conn.laddr.port == 8000 and conn.pid:
        try:
            server_process = psutil.Process(conn.pid)
            break
        except psutil.NoSuchProcess:
            pass

if server_process is None:
    print("Could not find FastAPI process on port 8000.")
    print("Start the server first.")
    raise SystemExit(1)

print("Server PID:", server_process.pid)
print(
    "Initial memory:",
    round(server_process.memory_info().rss / 1024 / 1024, 2),
    "MB",
)

times = []
memories = []
success = 0
failed = 0

for i in range(TOTAL_REQUESTS):
    try:
        before = server_process.memory_info().rss / 1024 / 1024

        start = time.perf_counter()

        with open(PDF, "rb") as f:
            response = requests.post(
                URL,
                files={
                    "file": (
                        os.path.basename(PDF),
                        f,
                        "application/pdf",
                    )
                },
                timeout=120,
            )

        elapsed = time.perf_counter() - start

        after = server_process.memory_info().rss / 1024 / 1024

        times.append(elapsed)
        memories.append(after)

        if response.status_code == 200:
            success += 1
        else:
            failed += 1

        print(
            f"Request {i + 1}: "
            f"status={response.status_code}, "
            f"time={elapsed:.2f}s, "
            f"memory_before={before:.2f}MB, "
            f"memory_after={after:.2f}MB, "
            f"delta={after - before:+.2f}MB"
        )

    except Exception as e:
        failed += 1
        print(f"Request {i + 1} failed:", e)

print("\n========== RESULT ==========")
print("Total requests :", TOTAL_REQUESTS)
print("Successful     :", success)
print("Failed         :", failed)

if times:
    print("Average time   :", round(sum(times) / len(times), 2), "sec")
    print("Max time       :", round(max(times), 2), "sec")

if memories:
    print("Initial memory :", round(memories[0], 2), "MB")
    print("Final memory   :", round(memories[-1], 2), "MB")
    print(
        "Memory growth  :",
        round(memories[-1] - memories[0], 2),
        "MB",
    )