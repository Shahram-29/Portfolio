# Car Sales SQL Analytics

Database-management assignment over a regional car-sales schema (`sellers`, `cars`, `regions`).

- **QS1** — top sale per region via a derived-table join on `MAX(price)`
- **QS2** — CTE joining sellers to car categories; categories whose average price beats the overall average (`HAVING`)
- **QS3** — `CASE` tiering into high-value / standard-value buckets with totals
- **QS4** — `LEFT JOIN` + count to find the least-served region

Files: `car_sales_queries.sql`, `DBMS_Report.pdf` (write-up; a recorded walkthrough accompanied the submission).
