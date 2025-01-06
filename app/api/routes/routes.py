from fastapi import FastAPI, Depends, HTTPException, Query, Header, status
from sqlalchemy.orm import Session
from typing import Annotated, Union, Dict
import logging
from dotenv import load_dotenv
from fastapi.security import OAuth2PasswordBearer
import os

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

load_dotenv()

#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dependency to get the DB session
def get_db():
    db = None
    try:
        yield db
    finally:
        db.close()

from fastapi.security import OAuth2PasswordRequestForm

@app.post('/login', response_model=Dict)
def login(
        payload: OAuth2PasswordRequestForm = Depends(),
        session: Session = Depends(get_db)
    ):
    """Processes user's authentication and returns a token
    on successful authentication.

    request body:

    - username: Unique identifier for a user e.g email, 
                phone number, name

    - password:
    """
    try:
        assert payload.username
        assert payload.username in os.getenv("USERS")
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user credentials"
        )

    return {
        "access_token": payload.username
    }
