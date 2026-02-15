-- =============================================
-- 建立有效庫存池
-- =============================================
IF OBJECT_ID('tempdb..#Temp_ActiveStock') IS NOT NULL
    DROP TABLE #Temp_ActiveStock

SELECT ItemCode, SUM(ISNULL(Qty, 0)) AS AvailableQty
INTO #Temp_ActiveStock
FROM T_Stock
WHERE Status = 'A' AND Dept != 'Sales'
GROUP BY ItemCode;

-- =============================================
-- 獲取最新定價
-- =============================================
IF OBJECT_ID('tempdb..#Temp_LatestPrice') IS NOT NULL
    DROP TABLE #Temp_LatestPrice

SELECT ItemCode, Price
INTO #Temp_LatestPrice
FROM T_Price
WHERE Date = '2026-02-15';

-- =============================================
-- 結算總價值 (直接 SELECT INTO，無 IF OBJECT_ID)
-- =============================================
SELECT s.ItemCode, (s.AvailableQty * p.Price) AS Total_Value
INTO #Output_Report
FROM #Temp_ActiveStock s
JOIN #Temp_LatestPrice p ON s.ItemCode = p.ItemCode;
