import {useRouter} from "next/router";
import {useEffect, useState} from "react";
import {Card, Container} from "react-bootstrap";
import Link from "next/link"
import ResultContent from "../components/ResultConent";
import ReactLoading from 'react-loading'
import axios from "axios";
import {isAssetError} from "next/dist/client/route-loader";
import MetricTable from "../components/MetricTable";

const _ = require('lodash')

export default function result() {

    const router = useRouter();

    const [apiQuery, setApiQuery] = useState();
    // const [isLoading, setIsLoading] = useState(true);
    const [resultData, setResultData] = useState([]);
    const [stockList, setStockList] = useState([]);
    const [errMsg, setErrMsg] = useState(undefined);

    // listen if the router is ready
    useEffect(() => {
        setApiQuery(router.query)
        if(router.isReady && _.isEmpty(router.query)) {
            setErrMsg("Invalid evaluation parameter was set.")
        }
    }, [router.isReady])

    // listen if the apiQuery is ready
    useEffect(() => {
        if (typeof apiQuery !== "undefined" && !_.isEmpty(apiQuery)){
            // TODO: fetch evaluation result from backend
            axios.get(
                process.env.NEXT_PUBLIC_EVALUTE_URL,
                {
                    params: apiQuery
                }
            ).then(function (res) {
                // if (res.status === 200){
                //
                //
                // }
                const tmpResultData = res.data

                setResultData(tmpResultData)
                setStockList([...new Set(
                    tmpResultData.map(x => x['stock'])
                )])
            }).catch((err) => {
                if (err.response?.data?.error){
                    setErrMsg(err.response.data.error)
                }
                else {
                    setErrMsg(`${err.message}, possibly evaluation parameter errors or server is not started.`)
                }
                console.log(err)
            })



            // load dummy json for dev, disable it when dev finished
            // const tmpResultData = require("../public/demo_evaluation_data.json");
            // setResultData(tmpResultData)
            // setStockList([...new Set(
            //     tmpResultData.map(x => x['stock'])
            // )])


            // console.log(apiQuery)
        }
    }, [apiQuery])

    return (
        <>
            <Container className={"d-flex flex-column min-vh-100 justify-content-center align-items-center"}>
                <Card className={"w-100 mx-auto hover-box"}>
                    <Card.Body>
                        <Card.Title as={"h3"}>
                            <Link href={"/"}>
                                <a className={"link-dark hover-underline"}><span>&#60;</span> Model Evaluation - Result</a>
                            </Link>
                        </Card.Title>
                        <Card.Subtitle className={"text-muted"}>A Comparative Study of BGRU and GAN for Stock Market
                            Forecasting in dual regions</Card.Subtitle>
                        <hr className={"mt-4"}/>
                        <div>
                            {/* Display loading indicator while waiting for the result*/}
                            {
                                (_.isEmpty(stockList) && _.isEmpty(resultData) && _.isEmpty(errMsg) ) && (
                                    <div className={"mt-4 text-center"}>
                                        <ReactLoading
                                            type={"bars"}
                                            color={"#000000"}
                                            height={"5vh"}
                                            width={"5vw"}
                                            className={"mx-auto"}
                                        />
                                        <br/>
                                        Evaluating...
                                    </div>
                                )
                            }

                            {/* Display evaluation result with plots*/}
                            {
                                (!_.isEmpty(stockList) &&
                                !_.isEmpty(resultData) &&
                                _.isUndefined(errMsg)) && (
                                    <>
                                        {stockList.map(x => (
                                            <ResultContent
                                                stock_to_display={x}
                                                evaluation_result={resultData}
                                            />
                                        ))}

                                        <hr className={"mt-4"}/>

                                        {/*TODO: Add summary table*/}
                                        <h2 className={"mt-2 mb-4"}>Model Summary</h2>
                                        <MetricTable
                                            data={resultData}
                                        />
                                    </>
                                )
                            }

                            {/* Display error message if backend throws it, hide the evaluation result part */}
                            {
                                !_.isUndefined(errMsg) && (
                                    <div className={"text-center"}>
                                        <p className={"text-danger"}>
                                            Error was reported from the server:<br/>
                                            {errMsg}
                                        </p>
                                    </div>
                                )
                            }

                        </div>
                    </Card.Body>
                </Card>
            </Container>
        </>
    )
}