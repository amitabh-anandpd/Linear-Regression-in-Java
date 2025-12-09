import java.util.List;
public class Test {

    public static void main(String[] args) {

        // ---------------------------
        // 1. Create dataset manually
        // ---------------------------
        List<List<Object>> data = new java.util.ArrayList<>();

        data.add(java.util.Arrays.asList(1.0, "A", 10.0));
        data.add(java.util.Arrays.asList(2.0, "B", 15.0));
        data.add(java.util.Arrays.asList(3.0, "A", 14.0));
        data.add(java.util.Arrays.asList(4.0, "B", 19.0));
        data.add(java.util.Arrays.asList(5.0, "A", 18.0));

        List<String> columns = java.util.Arrays.asList("x", "category", "y");

        DataFrame df = new DataFrame(data, columns);

        // ---------------------------
        // 2. Train Linear Regression
        // ---------------------------
        LinearRegression lr = new LinearRegression();
        lr.learningRate = 0.05;
        lr.epochs = 2000;

        // y is last column
        lr.fit(df, 2);

        // ---------------------------
        // 3. Test batch prediction
        // ---------------------------
        DataFrame df2 = df.deepCopy();
        df2.removeColumn(2);
        System.out.println("\nBatch Predictions:");
        double[] preds = lr.predict(df2);

        for (int i = 0; i < preds.length; i++) {
            System.out.printf("Pred[%d] = %.3f%n", i, preds[i]);
        }

        // ---------------------------
        // 4. Test single-row prediction
        // ---------------------------
        System.out.println("\nSingle Row Test:");
        List<Object> testRow = java.util.Arrays.asList(6.0, "B");
        double pred = lr.predict(testRow);

        System.out.println("Input: x=6, category=B");
        System.out.println("Predicted y: " + pred);
        System.out.println("Expected y approx: " + (2*6 + 3*1 + 5));
    }
}
