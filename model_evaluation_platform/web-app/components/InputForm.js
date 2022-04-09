import {Formik} from 'formik'
import {Button, Col, Form, Row} from "react-bootstrap";
import * as yup from 'yup'
import CreatableSelect from "react-select/creatable";
import {createRef, useEffect, useState} from "react";
import axios from 'axios';
// import "flatpickr/dist/themes/light.css"
import "flatpickr/dist/flatpickr.min.css"
import Flatpickr from "react-flatpickr"
import moment from 'moment'

const schema = yup.object().shape({
    // dropdown combo box (multi-select) field with react-select
    stocks: yup.array()
        .min(1, 'Please select at least 1 stock')
        .of(
            yup.object().shape({
                label: yup.string().required(),
                value: yup.string().required(),
            })
        ),
    // start date field
    start_date: yup.string().required("Please select a start date"),
    // end date field
    end_date: yup.string().required("Please select an end date"),
})

export default function InputForm(props) {

    const {
        onSubmit
    } = props

    const [defaultStocksOption, setStocksOption] = useState();

    const datePickerRef = createRef();


    useEffect(() => {
        // fetch default stocks data from server
        axios.get(process.env.NEXT_PUBLIC_DEFAULT_STOCKS_URL)
            .then(function (res) {
                const {default_stocks} = res.data;
                const processed_data = default_stocks.map((x) => {
                    return {
                        label: x.toUpperCase(),
                        value: x.toUpperCase()
                    }
                })

                setStocksOption(processed_data)
            })

    }, [])


    return (
        <>
            <Formik
                validationSchema={schema}
                onSubmit={onSubmit}
                initialValues={{
                    stocks: [],
                    start_date: moment().subtract(1, 'years').format("YYYY-MM-DD"),
                    end_date: moment().format("YYYY-MM-DD"),
                }}
            >
                {({
                      handleSubmit,
                      handleReset,
                      handleChange,
                      handleBlur,
                      values,
                      initialValues,
                      touched,
                      isValid,
                      errors,
                      setFieldValue,
                      setTouched,
                  }) => (
                    <Form noValidate onSubmit={handleSubmit} onReset={handleReset}>
                        <Row className={"mb-3"}>
                            <Col>
                                Targeted models: <br/> BGRU (HK), BRU(US), GAN (HK), GAN (US)
                            </Col>
                        </Row>
                        <Row className={"mb-3"}>
                            <Form.Group as={Col} controlId={"evaluation_stocks"}>
                                <Form.Label>Stock Symbols for evaluation</Form.Label>
                                <CreatableSelect
                                    instanceId={"stock_symbol-react-select"}
                                    isMulti
                                    value={values.stocks}
                                    onChange={(value) => setFieldValue("stocks", value)}
                                    options={defaultStocksOption}
                                />
                                {touched.stocks && !!errors.stocks && (
                                    <span className={"text-danger"}>
                                        {errors.stocks}
                                    </span>
                                )}
                            </Form.Group>
                        </Row>
                        <Row className={"mb-3"}>
                            <Form.Group as={Col} controlId={"start_date"}>
                                <Form.Label>Stock data start date</Form.Label>
                                {/*TODO: use react select to choose stocks, allow custom input*/}
                                <Flatpickr
                                    options={{
                                        maxDate: "today"
                                    }}
                                    defaultValue={initialValues.start_date}
                                    className={"d-block w-100 form-control bg-white"}
                                    onChange={([date]) => {
                                        setFieldValue("start_date",
                                            typeof date !== 'undefined' ? moment(date).format("YYYY-MM-DD")
                                                : date
                                        )
                                        setTouched("start_date", true)
                                    }}
                                    value={values.start_date}
                                />
                                {!!errors.start_date && (
                                    <span className={"text-danger"}>
                                        {errors.start_date}
                                    </span>
                                )}
                            </Form.Group>
                        </Row>
                        <Row className={"mb-3"}>
                            <Form.Group as={Col} controlId={"end_date"}>
                                <Form.Label>Stock data end date <span className={"text-muted"}>(At least 30 trading days from the start date)</span></Form.Label>
                                {/*TODO: use react select to choose stocks, allow custom input*/}
                                <Flatpickr
                                    options={{
                                        maxDate: "today"
                                    }}
                                    defaultValue={initialValues.end_date}
                                    className={"d-block w-100 form-control bg-white"}
                                    onChange={([date]) => {
                                        setFieldValue("end_date",
                                            typeof date !== 'undefined' ? moment(date).format("YYYY-MM-DD")
                                                : date
                                        )
                                        setTouched("end_date", true)
                                    }}
                                    value={values.end_date}
                                />
                                {!!errors.end_date && (
                                    <span className={"text-danger"}>
                                        {errors.end_date}
                                    </span>
                                )}
                            </Form.Group>
                        </Row>

                        <div className={"text-end"}>
                            <Button variant={"outline-warning"} type={"reset"}
                                    className={"align-right mx-1"}>Clear</Button>
                            <Button type={"submit"} className={"align-right mx-1"}>Evaluate</Button>
                        </div>
                    </Form>
                )}
            </Formik>
        </>
    )
}