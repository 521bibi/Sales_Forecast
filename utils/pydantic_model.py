import datetime
from typing import Any, Mapping, Optional
from pydantic import BaseModel


# API-1: API-销售预测byProduct
# API-1【请】: API-销售预测byProduct   店ID , 产品ID
class Store_product_id(BaseModel):
    store_id: str     #
    product_id: str      #


# API-1【中】: API-销售预测byProduct   日期，店ID , 产品ID，预测值
class response_by_Product(BaseModel):
    SDATE: Optional[str]   #
    HOLIDAY: Optional[str]      #
    CELEBRATION: Optional[str]
    STOREID: str
    PRODUCTID: str
    LOW_TEMP: Optional[str]
    HIGH_TEMP: Optional[str]
    SKY: Optional[str]
    ENTRY: Optional[str]
    EXIT: Optional[str]
    ONSALE: Optional[str]
    COVID19: Optional[str]
    predict: float


# API-1【返】: API-销售预测byProduct   单一产品未来7日
class Forecast7D_by_product(BaseModel):
    first:   response_by_Product
    second:  response_by_Product
    third:   response_by_Product
    fourth:  response_by_Product
    fifth:   response_by_Product
    sixth:   response_by_Product
    seventh: response_by_Product


# API-1【返】: API-销售预测byProduct   单一产品未来7日
class Forecast(BaseModel):
    __root__: Any
