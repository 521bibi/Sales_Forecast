import json

import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from inference import InferencebyProduct
from utils.crud import connect_mongo, findFromMongoDB
from utils.pydantic_model import Forecast7D_by_product, Forecast

tags_metadata = [
    {
        "name": "API-销售预测byProduct",
        "description": "Sales_Forecast 销售预测byProduct",
    },
    {
        "name": "API-销售预测byStore",
        "description": "Sales_Forecast",
    },
]

app = FastAPI(
    title="My Sales_Forecast Project",
    description="Sales_Forecast API",
    version="0.1.0",
    openapi_tags=tags_metadata
)
router = APIRouter()


@router.get('/')
def root():
    return {'查看接口文档': 'http://192.168.8.68:15300/docs'}


@router.post('/forecastbyProduct/', response_model=Forecast, tags=['API-销售预测byProduct'])
def ForecastbyProduct(store_id: str, product_id: str):
    parsed = {'提示信息': '无此产品或暂未对此产品建立预测模型'}
    try:
        # store_id, product_id = '12', '141691'
        test = InferencebyProduct(store_id, product_id)
        x = test.forecasting_out()
        out = x.to_json(orient="index", force_ascii=False)
        parsed = json.loads(out)
        print(out)
    except Exception as r:
        print('未知错误 %s' % r)

    return parsed


app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == '__main__':
    link_collection = 'FIRecommend_task'
    set1 = connect_mongo(link_collection)
    print(findFromMongoDB(set1))
    uvicorn.run(app=app, host="0.0.0.0", port=15300, workers=1)
