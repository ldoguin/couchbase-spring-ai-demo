package com.example.demo;

import com.couchbase.client.java.Cluster;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.*;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.core.annotation.Order;

import java.util.List;

@SpringBootApplication
public class DemoApplication {

    private static final Logger logger = LoggerFactory.getLogger(DemoApplication.class);

    public static void main(String[] args) {
        SpringApplication app = new SpringApplication(DemoApplication.class);
        app.setWebApplicationType(WebApplicationType.NONE);
        ConfigurableApplicationContext ctx = app.run(args);

	}

    @Bean
    VectorStore colorVectorStore(Cluster cluster){
        CouchbaseVectorStore.CouchbaseVectorStoreConfig config = CouchbaseVectorStore.CouchbaseVectorStoreConfig
            .builder()
            .withBucketName("ai_color")
            .withScopeName("ai_color")
            .withCollectionName("ai_color")
            .withIndexOptimization(CouchbaseIndexOptimization.recall)
            .withSimilarityFunction(CouchbaseSimilarityFunction.l2_norm)
            .withDimensions(3)
            .withVectorIndexName("ai_color_index")
            .build();
        return new CouchbaseVectorStore(cluster, new JsonStringEmbeddingModel(), config, true);
    }

    @Bean
    @Order(Integer.MAX_VALUE)
    public CommandLineRunner searchAdjacentColor(JsonDocLoader jsonDocLoader, VectorStore vectorStore, VectorStore colorVectorStore) {
        return args -> {
            logger.info("Loading docs...");
            jsonDocLoader.run();
            logger.info("Docs loaded");
            logger.info("What are the color closest to [0.0,0.0,128.0] ?");
            List<Document> similarColors = colorVectorStore.similaritySearch("[0.0,0.0,128.0]");
            similarColors.forEach(s -> logger.info(s.toString()));
            String question = "What is the color that is often associated with luxury and elegance, and is a combination of deepening and enriching the color red?";
            SearchRequest searchRequest = SearchRequest.defaults()
                .withQuery(question).withTopK(1);
            similarColors = vectorStore.similaritySearch(searchRequest);
            logger.info(similarColors.get(0).toString());

        };
    };

}
