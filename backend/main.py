from fastapi import FastAPI
from settings import config
from fastapi.middleware.cors import CORSMiddleware

from api import router

app = FastAPI(
    title="TripRecs",
    docs_url=f"{config.API_STR}/docs",
    openapi_url=f"{config.API_STR}/openapi.json"
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=config.CORS_ORIGINS_LIST,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.include_router(router=router, prefix=config.API_STR)


