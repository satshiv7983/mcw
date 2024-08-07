
import asyncio
import random,os
import time

async def create_random(size=5):
    return os.urandom(size).hex()


async def random_sleep(caller):
    t=random.randint(0,5)
    if caller:
        print(f"{caller} is sleeping for {t} sec")
    await asyncio.sleep(t)


async def produce(name,queue):
    for i in range(2):
        await random_sleep(f'Producer :{name}')
        r=await create_random()
        t=time.perf_counter()
        await queue.put((r,t))
        print(f"Producer {name} has pushed item {r} in queue")

async def consume(name,queue):
    while True:
        await random_sleep(f"Consumer : {name}")
        i,t=await queue.get()
        now=time.perf_counter()
        print(f"Consumer {name} got element <{i}>"
              f" in {now-t:0.5f} seconds.")
        queue.task_done()

async def main(pnums,cnums):
    random.seed(33)
    queue=asyncio.Queue()
    
    producers=[asyncio.create_task(produce(n,queue)) for n in range(pnums)]
    consumers=[asyncio.create_task(consume(n,queue)) for n in range(cnums)]

    await asyncio.gather(*producers)

    await queue.join()

    for c in consumers:
        c.cancel()


if __name__=="__main__":
    asyncio.run(main(pnums=2,cnums=2))




# when not using random_sleep
# when not awaiting queue.join()