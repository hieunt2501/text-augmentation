import logging
import requests
from queue import Queue

from app.core.config import MAX_CACHE_SIZE


class BaseService:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_cache_size = MAX_CACHE_SIZE
        self.max_retry = 5
        self.cache = {}
        self.cache_queue = Queue(maxsize=self.max_cache_size)
        self.session = None
        self.logger.info(f"CREATE {self.__class__.__name__}")

    def init_session(self, force=False):
        if self.session is None or force:
            self.session = requests.session()

    def call_back_func(self):
        self.init_session(True)

    def make_request(self, request_func, call_back_func=None, key=None):
        if key is not None and key in self.cache:
            self.logger.info(f"GET key [{key}] from cache [{self.__class__.__name__}]")
            return self.cache[key]

        self.init_session(False)
        result = None
        retry = 0
        while result is None:
            try:
                result = request_func()
            except Exception as e:
                if retry == self.max_retry:
                    self.logger.warning("Max retry exceed")
                    break
                retry += 1
                self.logger.error(f"Cannot make request. Error : {e}")
                if call_back_func is not None:
                    call_back_func()
                else:
                    self.call_back_func()

        if key is not None:
            self.cache[key] = result
            self.cache_queue.put(key)

            if self.cache_queue.qsize() == self.max_cache_size:
                remove_key = self.cache_queue.get()
                if remove_key in self.cache:
                    del self.cache[remove_key]
                    self.logger.info(f"REMOVE key [{key}] from cache [{self.__class__.__name__}]")
        return result
