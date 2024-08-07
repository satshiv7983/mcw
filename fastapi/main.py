from fastapi import FastAPI,HTTPException
import asyncio,argparse,uvicorn

app = FastAPI()

@app.get("/add")
async def add(a: float, b: float):
    await asyncio.sleep(2)  
    return {"result": a + b}

@app.get("/subtract")
async def subtract(a: float, b: float):
    await asyncio.sleep(0) 
    return {"result": a - b}

@app.get("/multiply")
async def multiply(a: float, b: float):
    await asyncio.sleep(3) 
    return {"result": a * b}

@app.get("/divide")
async def divide(a: float, b: float):
    if b == 0:
        raise HTTPException(status_code=400, detail="Division by zero is not allowed")
    await asyncio.sleep(0)  
    return {"result": a / b}


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Run FastAPI server with Uvicorn.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to.")
    parser.add_argument("--workers", type=int, default=10, help="Number of worker processes.")

    args = parser.parse_args()

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        workers=args.workers
    )
