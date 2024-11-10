from hashlib import sha256
from fastapi import UploadFile


async def get_sha256_hash(file: UploadFile) -> str:
    hash = sha256()
    
    await file.seek(0)

    while True:
        chunk = await file.read(1024 * 1024)

        if not chunk:
            break

        hash.update(chunk)

    await file.seek(0)

    return hash.hexdigest()

