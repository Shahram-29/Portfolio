--QS1
SELECT first_name, last_name, selling_date, price
FROM sellers AS s
INNER JOIN (SELECT region_id , MAX(price) AS max_price
FROM sellers
GROUP BY region_id) AS max_table
ON s.region_id = max_table.region_id
AND s.price = max_table.max_price
--QS2 
WITH prices AS
( SELECT c.category, s.price
 FROM sellers AS s
 JOIN cars AS c ON s.car = c.car )
SELECT category, AVG(price) AS avg_price
FROM prices
GROUP BY category
HAVING AVG(price) > (SELECT AVG (price) FROM prices)
ORDER BY avg_price DESC;
--QS3
SELECT 
  CASE 
    WHEN price >= 50000 THEN 'High-Value' 
    ELSE 'Standard-Value' 
  END AS price_category,
  SUM(price) AS total_value
FROM sellers
GROUP BY 
  CASE 
    WHEN price >= 50000 THEN 'High-Value' 
    ELSE 'Standard-Value' 
  END;
--QS4
SELECT r.region_id, r.region, COUNT(s.seller_id) AS car_count
FROM regions r
LEFT JOIN sellers s ON r.region_id = s.region_id
GROUP BY r.region_id, r.region
ORDER BY car_count
LIMIT 1;

