from fastapi import APIRouter

router = APIRouter()

@router.get("/", tags=["TripRecs"])
async def home():
    return {
        "message": f"TripRecs Backend"
    }