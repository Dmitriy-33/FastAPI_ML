import logging
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

logger = logging.getLogger(__name__)
security = HTTPBasic()

USERS = {"admin": "secret123"}


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    ok_user = secrets.compare_digest(
        credentials.username, next(iter(USERS), "")
    )
    ok_pass = secrets.compare_digest(
        credentials.password, USERS.get(credentials.username, "")
    )

    if not (ok_user and ok_pass):
        logger.warning("Неудачная попытка входа: %r", credentials.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username
