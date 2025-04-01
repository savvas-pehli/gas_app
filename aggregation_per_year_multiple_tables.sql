DO $$
DECLARE
    tables text[] := ARRAY[{{tables}}]; -- Your tables here
    tbl text;
    all_columns text[];
    numeric_columns text[];
    common_columns text[];
    col text;
    query text := '';
    union_needed boolean := false;
    group_by_columns text[];
    has_date boolean;
BEGIN
    -- Get all columns from all tables
    SELECT array_agg(DISTINCT column_name::text) INTO all_columns
    FROM information_schema.columns
    WHERE table_schema = 'public'
    AND table_name = ANY(tables);
    
    -- Get all numeric columns
    SELECT array_agg(DISTINCT column_name::text) INTO numeric_columns
    FROM information_schema.columns
    WHERE table_schema = 'public'
    AND table_name = ANY(tables)
    AND data_type IN ('numeric', 'integer', 'bigint', 'real', 'double precision');
    
    -- Process each table
    FOREACH tbl IN ARRAY tables
    LOOP
        -- Check if table has Date column
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = tbl 
            AND column_name = 'Date'
        ) INTO has_date;
        
        -- Get columns that exist in this specific table
        SELECT array_agg(column_name::text) INTO common_columns
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND table_name = tbl;
        
        IF union_needed THEN
            query := query || ' UNION ALL ';
        ELSE
            union_needed := true;
        END IF;
        
        -- Start building this table's query
        query := query || format('
            SELECT 
                %L AS source_table', tbl);
        
        -- Add year if Date exists
        IF has_date THEN
            query := query || ',
                EXTRACT(YEAR FROM DATE_TRUNC(''year'', "Date")) AS year';
        ELSE
            query := query || ',
                NULL::integer AS year';
        END IF;
        
        -- Initialize GROUP BY columns for this table
        group_by_columns := ARRAY[]::text[];
        IF has_date THEN
            group_by_columns := array_append(group_by_columns, 'DATE_TRUNC(''year'', "Date")');
        END IF;
        
        -- Add all columns (numeric and non-numeric)
        FOREACH col IN ARRAY all_columns
        LOOP
            CONTINUE WHEN col = 'Date' OR col = 'Hour'; -- Skip Date (handled) and Hour (not needed)
            
            IF col = ANY(common_columns) THEN
                IF col = ANY(numeric_columns) THEN
                    -- Numeric column - aggregate
                    query := query || format(',
                    AVG("%s") FILTER (WHERE "%s" IS NOT NULL) AS "%s"', col, col, col);
                ELSE
                    -- Non-numeric column - include in GROUP BY
                    query := query || format(',
                    "%s"', col);
                    group_by_columns := array_append(group_by_columns, format('"%s"', col));
                END IF;
            ELSE
                -- Column doesn't exist in this table - use NULL
                IF col = ANY(numeric_columns) THEN
                    query := query || format(',
                    NULL::numeric AS "%s"', col);
                ELSE
                    query := query || format(',
                    NULL AS "%s"', col);
                END IF;
            END IF;
        END LOOP;
        
        -- Complete the table's query with GROUP BY
        query := query || format('
            FROM %I', tbl);
            
        IF array_length(group_by_columns, 1) > 0 THEN
            query := query || format('
            GROUP BY %s', array_to_string(group_by_columns, ', '));
        END IF;
    END LOOP;
    
    -- Create the final view
    EXECUTE 'DROP VIEW IF EXISTS multi_table_yearly_aggregation;
CREATE VIEW  multi_table_yearly_aggregation AS 
             ' || query || ' 
             ORDER BY source_table, year';
    
    RAISE NOTICE 'View created: multi_table_yearly_aggregation';
    RAISE NOTICE 'Query: %', query;
END $$;

-- Query the results
SELECT * FROM multi_table_yearly_aggregation;