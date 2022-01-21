import {useTable, useSortBy} from 'react-table';
import {Table} from "react-bootstrap";
import {useMemo} from "react";


export default function MetricTable(props) {
    const {
        // columns,
        data
    } = props;

    const columns = useMemo(() => [
        {
            Header: "model",
            accessor: "model",
        },
        {
            Header: "stock",
            accessor: "stock",
        },
        {
            Header: "Loss (MSE)",
            accessor: "loss (mean_squared_error)",
            sortType: "basic",
        },
        {
            Header: "MAE",
            accessor: "mean_absolute_error",
            sortType: "basic",
        },
        {
            Header: "RMSE",
            accessor: "root_mean_squared_error",
            sortType: "basic",
        },
        {
            Header: "MAPE",
            accessor: "mean_absolute_percentage_error",
            sortType: "basic",
        }
    ], [])

    const {
        getTableProps,
        getTableBodyProps,
        headerGroups,
        rows,
        prepareRow,
    } = useTable(
        {
            columns,
            data
        },
        useSortBy
    )

    return (
        <>
            <Table responsive striped bordered hover {...getTableProps()}>
                <thead>
                {headerGroups.map(headerGroups => (
                    <tr {...headerGroups.getHeaderGroupProps()}>
                        {headerGroups.headers.map(column => (
                            <th {...column.getHeaderProps(column.getSortByToggleProps())}>
                                {column.render('Header')}
                                {/* Sort direction indicator */}
                                <span className={"fs-5"}>
                                    {column.isSorted
                                        ? column.isSortedDesc
                                            ? ' \u2193'
                                            : " \u2191"
                                        : ' \u296F'}
                                </span>
                            </th>
                        ))}
                    </tr>
                ))}
                </thead>
                <tbody {...getTableBodyProps()}>
                    {rows.map((row, i) => {
                        prepareRow(row);
                        return (
                            <tr {...row.getRowProps()}>
                                {row.cells.map((cell) => {
                                    return <td {...cell.getCellProps()}>{cell.render("Cell")}</td>;
                                })}
                            </tr>
                        )
                    })}
                </tbody>
            </Table>
        </>
    )
}