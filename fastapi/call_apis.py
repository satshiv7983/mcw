import asyncio
import httpx
import time

base_url = "http://127.0.0.1:8000"

async def add(client, a, b):
    response = await client.get(f"{base_url}/add", params={"a": a, "b": b})
    result = response.json()
    print(f"Addition: {a} and {b} = {result['result']}")

async def subtract(client, a, b):
    response = await client.get(f"{base_url}/subtract", params={"a": a, "b": b})
    result = response.json()
    print(f"Subtraction: {a} and {b} = {result['result']}")


async def multiply(client, a, b):
    response = await client.get(f"{base_url}/multiply", params={"a": a, "b": b})
    result = response.json()
    print(f"Multiplication: {a} and {b} = {result['result']}")


async def divide(client, a, b):
    response = await client.get(f"{base_url}/divide", params={"a": a, "b": b})
    result = response.json()
    print(f"Division: {a} and {b} = {result['result']}")

async def main():
    timeout = httpx.Timeout(10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        a, b = 10, 5

        start_time = time.time()
        
        tasks = [
            add(client, a, b),
            subtract(client, a, b),
            multiply(client, a, b),
            divide(client, a, b)
        ]

        # tasks=[add(client, a, b) for i in range(100)]
        
        await asyncio.gather(*tasks)  # Await all tasks
        
        end_time = time.time()
        duration = end_time - start_time

        print(f"Time taken: {duration:.2f} sec")

if __name__ == "__main__":
    asyncio.run(main())
