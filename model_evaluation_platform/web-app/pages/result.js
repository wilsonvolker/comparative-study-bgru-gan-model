import {useRouter} from "next/router";
import {useEffect, useState} from "react";
import {Card, Container} from "react-bootstrap";
import Link from "next/link"

const _ = require('lodash')

export default function result() {

    const router = useRouter();

    const [apiQuery, setApiQuery] = useState();
    const [isLoading, setIsLoading] = useState(true);

    // listen if the router is ready
    useEffect(() => {
        setApiQuery(router.query)
    }, [router.isReady])

    // listen if the apiQuery is ready
    useEffect(() => {
        if (typeof apiQuery !== "undefined" && !_.isEmpty(apiQuery)){
            // TODO: fetch evaluation result from backend
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

                        <div className={"mt-4"}>
                            RESULT HERE
                            {/*TODO: Display evaluation result with plots*/}

                            {/*TODO: Display error message if backend throws it, hide the evaluation result part */}
                        </div>
                    </Card.Body>
                </Card>
            </Container>
        </>
    )
}