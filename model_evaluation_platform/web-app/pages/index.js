import Head from 'next/head'
import Image from 'next/image'
import styles from '../styles/Home.module.css'
import {Card, Container} from "react-bootstrap";
import InputForm from "../components/InputForm";
import moment from 'moment'
import {useRouter} from "next/router";

export default function Home() {
    const router = useRouter()

    const onSubmit = function(values, formikBag) {
        // console.log("submitted")
        // console.log(values)

        // Convert all stock symbols to upper case
        let stocks = values.stocks.map(x => x.value.toUpperCase()).join(',');

        router.push({
            pathname: "/result",
            query: {
                stocks: stocks,
                start_date: values.start_date,
                end_date: values.end_date,
            }
        })
        // formikBag.setSubmitting(false)
    }

    return (
        <>
            <Head>
                <title>Model Evaluation Platform - A Comparative Study of BGRU and GAN for Stock Market Forecasting in dual regions</title>
            </Head>
            <Container className={"d-flex flex-column min-vh-100 justify-content-center align-items-center"}>
                <Card className={"mw-100 mw-lg-50 mx-auto hover-box"}>
                    <Card.Body>
                        <Card.Title as={"h3"}>Model Evaluation Platform</Card.Title>
                        <Card.Subtitle className={"text-muted"}>A Comparative Study of BGRU and GAN for Stock Market
                            Forecasting in dual regions</Card.Subtitle>
                        {/*<Card.Text className={"mt-4"}>*/}

                        {/*</Card.Text>*/}
                        <div className={"mt-4"}>
                            <InputForm
                                onSubmit={onSubmit}
                            />
                        </div>
                    </Card.Body>
                </Card>
            </Container>
        </>
    )
}
