
import concurrent.futures
import requests
import time

URL = "http://localhost:8001/predict"

PAYLOAD = {
    "amount": 1500,
    "transaction_type": "TRANSFER",
    "origin_balance_before": 5000,
    "destination_balance_before": 2000,
    "step": 100
}

def send_request(_):
    try:
        response = requests.post(URL, json=PAYLOAD, timeout=10)
        return response.status_code
    except requests.RequestException as error:
        return str(error)

if __name__ == "__main__":
    total_requests = 500
    workers = 10

    print(f"Sending {total_requests} requests using {workers} workers...")

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(send_request, range(total_requests)))

    elapsed = time.time() - start

    successful = results.count(200)
    failed = total_requests - successful

    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Successful requests: {successful}")
    print(f"Failed requests: {failed}")
