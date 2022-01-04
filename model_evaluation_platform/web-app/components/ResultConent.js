import {useEffect, useState} from "react";


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

    const [filteredData, setFilteredData] = useState();

    useEffect(() => {
        const tmp_filteredData = evaluation_result.filter((x) => {
            return x["stock"].toUpperCase() === stock_to_display.toUpperCase()
        })

        setFilteredData(tmp_filteredData)
    }, [])

    return (
        <>
            <h4>
                {stock_to_display}
            </h4>
        </>
    )
}