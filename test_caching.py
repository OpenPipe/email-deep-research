import litellm
from litellm import acompletion
from litellm.caching.caching import Cache, LiteLLMCacheType
import asyncio

litellm.cache = Cache(type=LiteLLMCacheType.DISK)


async def main():
    # Make completion calls
    response1 = await acompletion(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke."}],
        caching=True,
    )
    response2 = await acompletion(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke."}],
        caching=True,
    )

    print(response1)
    print(response2)


if __name__ == "__main__":
    asyncio.run(main())
