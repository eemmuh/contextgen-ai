"""
Rate limiting middleware for the API.
"""

import time
import asyncio
from typing import Dict, Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from src.utils.logger import get_logger

logger = get_logger("rate_limiter")


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute per client
        """
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Use X-Forwarded-For for proxy support, fallback to client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _cleanup_old_requests(self):
        """Remove requests older than 1 minute."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        for client_id in list(self.requests.keys()):
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > minute_ago
            ]
            
            # Remove empty client entries
            if not self.requests[client_id]:
                del self.requests[client_id]
    
    async def _start_cleanup_task(self):
        """Start periodic cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Periodic cleanup loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Clean up every minute
                self._cleanup_old_requests()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def is_allowed(self, request: Request) -> bool:
        """
        Check if request is allowed based on rate limits.
        
        Args:
            request: FastAPI request object
            
        Returns:
            True if request is allowed, False otherwise
        """
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        # Initialize client if not exists
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Add current request
        self.requests[client_id].append(current_time)
        
        # Check if within rate limit
        recent_requests = [
            req_time for req_time in self.requests[client_id]
            if req_time > current_time - 60
        ]
        
        # Update client requests to only recent ones
        self.requests[client_id] = recent_requests
        
        return len(recent_requests) <= self.requests_per_minute
    
    def get_remaining_requests(self, request: Request) -> int:
        """Get remaining requests for client."""
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        if client_id not in self.requests:
            return self.requests_per_minute
        
        recent_requests = [
            req_time for req_time in self.requests[client_id]
            if req_time > current_time - 60
        ]
        
        return max(0, self.requests_per_minute - len(recent_requests))


# Global rate limiter instance
rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware for FastAPI.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware/endpoint function
        
    Returns:
        Response from next middleware/endpoint
    """
    # Start cleanup task if not running
    await rate_limiter._start_cleanup_task()
    
    # Check rate limit
    if not rate_limiter.is_allowed(request):
        remaining = rate_limiter.get_remaining_requests(request)
        
        logger.warning(
            f"Rate limit exceeded for client {rate_limiter._get_client_id(request)}"
        )
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after": 60,
                "remaining_requests": remaining
            },
            headers={
                "Retry-After": "60",
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Limit": str(rate_limiter.requests_per_minute)
            }
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    
    remaining = rate_limiter.get_remaining_requests(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.requests_per_minute)
    
    return response 