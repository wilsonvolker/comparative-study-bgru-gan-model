import {useRouter} from "next/router";
import {useEffect, useState} from "react";
import {Card, Container} from "react-bootstrap";
import Link from "next/link"
import ResultContent from "../components/ResultConent";

const _ = require('lodash')

export default function result() {

    const router = useRouter();

    const [apiQuery, setApiQuery] = useState();
    const [isLoading, setIsLoading] = useState(true);
    const [resultData, setResultData] = useState();

    // listen if the router is ready
    useEffect(() => {
        setApiQuery(router.query)
    }, [router.isReady])

    // listen if the apiQuery is ready
    useEffect(() => {
        if (typeof apiQuery !== "undefined" && !_.isEmpty(apiQuery)){
            // TODO: fetch evaluation result from backend

            // load dummy json for dev, TODO: Remove it when finished
            setResultData(require("../public/demo_evaluation_data.json"))

            console.log(apiQuery)
        }
    }, [apiQuery])

    return (
        <>
            <Container className={"d-flex flex-column min-vh-100 justify-content-center align-items-center"}>
                <Card className={"w-100 mx-auto hover-box"}>
                    <Card.Body>
                        <Card.Title as={"h3"}>
                            <Link href={"/"}>
                                <a className={"link-dark text-decoration-none"}><span>&#60;</span> Model Evaluation - Result</a>
                            </Link>
                        </Card.Title>
                        <Card.Subtitle className={"text-muted"}>A Comparative Study of BGRU and GAN for Stock Market
                            Forecasting in dual regions</Card.Subtitle>
                        <hr className={"mt-4"}/>
                        <div className={"mt-4"}>
                            {/*TODO: Display evaluation result with plots*/}
                            <ResultContent
                                stock_to_display={"AAPL"}
                                evaluation_result={resultData}
                            />

                            {/*TODO: Display error message if backend throws it, hide the evaluation result part */}
                        </div>
                    </Card.Body>
                </Card>
            </Container>
        </>
    )
}