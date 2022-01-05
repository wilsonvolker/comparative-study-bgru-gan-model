import {useEffect, useMemo, useState} from "react";
import {Col, Row, Table} from "react-bootstrap";
import { useTable, useSortBy } from 'react-table';
import MetricTable from "./MetricTable";
const _ = require('lodash')

export default function ResultContent(props) {
    /**
     * Example evaluation result array:
     * [
         {
            loss (mean_squared_error): 0.008053119294345379
            mean_absolute_error: 0.05010818690061569
            mean_absolute_percentage_error: 0.010865937095113127
            model: "BGRU (HK)"
            plot: "iVBORw0KGgoAAAANSUhEUgAAG1gAAAnECAYAAAADfuBQAAAAO..." // base64 decoded image string
            root_mean_squared_error: 0.08973917365074158
            stock: "1038.HK"
         }
     * ]
     */
    const {
        stock_to_display, // stock symbol
        evaluation_result // array of objects
    } = props

    const [filteredData, setFilteredData] = useState([]);

    useEffect(() => {
        if (!_.isEmpty(evaluation_result)) {
            const tmp_filteredData = evaluation_result.filter((x) => {
                return x["stock"].toUpperCase() === stock_to_display.toUpperCase()
            })

            setFilteredData(tmp_filteredData)
            console.log(tmp_filteredData)
        }
    }, [evaluation_result])

    const metricData = useMemo(() => filteredData, [filteredData])

    return (
        <div className={"mt-4"}>
            <h3>
                {stock_to_display}
            </h3>
            <Row>
                {filteredData.map(x => (
                    <Col lg={6} key={`${x['stock']} - ${x['model']}`}>
                        <img className={"img-fluid"} src={`data:image/png;base64, ${x['plot']}`} alt={`${x['stock']} - ${x['model']} stock price prediction result`}/>
                    </Col>
                ))}
            </Row>

            {/*TODO: Make a table to show the metrics of that stocks*/}
            <Row className={"mt-4"}>
                <MetricTable
                    data={metricData}
                />
            </Row>
            <Row>

            </Row>
        </div>
    )
}