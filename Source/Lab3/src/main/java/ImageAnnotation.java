import clarifai2.api.ClarifaiBuilder;
import clarifai2.api.ClarifaiClient;
import clarifai2.api.ClarifaiResponse;
import clarifai2.dto.input.ClarifaiInput;
import clarifai2.dto.input.image.ClarifaiImage;
import clarifai2.dto.model.output.ClarifaiOutput;
import clarifai2.dto.prediction.Concept;
import okhttp3.OkHttpClient;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.typography.hershey.HersheyFont;

import java.io.File;
import java.io.IOException;
import java.util.List;


public class ImageAnnotation {
    public static void main(String[] args) throws IOException {
        String filePath = "input/sample.mkv";
        KeyFrameDetection keyFrames = new KeyFrameDetection();
        keyFrames.getFrames(filePath);
        keyFrames.getMainFrames();
        final ClarifaiClient client = new ClarifaiBuilder("gkRbcRqTrqFxXIWg8oWqZf8FpnkHLXw81V_skWzY", "TsqyYSMhocLidZ1s-Q-wtFVmWEnP4PAt8p9O1iSI")
                .client(new OkHttpClient())
                .buildSync();
        client.getToken();

        File file = new File("output/mainframes");
        File[] files = file.listFiles();
        System.out.println("The input video, " + filePath + ", had " + files.length + " main scenes determined.  They contain the following themes: ");
        for (int i=0; i<files.length;i++){
            ClarifaiResponse response = client.getDefaultModels().generalModel().predict()
                    .withInputs(
                            ClarifaiInput.forImage(ClarifaiImage.of(files[i]))
                    )
                    .executeSync();
            List<ClarifaiOutput<Concept>> predictions = (List<ClarifaiOutput<Concept>>) response.get();
            MBFImage image = ImageUtilities.readMBF(files[i]);
            int x = image.getWidth();
            int y = image.getHeight();

            int sceneNumber = i+1;
            System.out.print("Scene " + sceneNumber + " had ");
            List<Concept> data = predictions.get(0).data();
            for (int j = 0; j < data.size(); j++) {
                System.out.print(data.get(j).name() + " ");
            }
            System.out.println();

        }

    }
}
