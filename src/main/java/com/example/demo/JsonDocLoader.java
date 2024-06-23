package com.example.demo;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Component
public class JsonDocLoader {


    @Value("classpath:color_data_2vectors/rgb_questions.json")
    private Resource rgbQuestionsResourceFile;

    @Value("classpath:color_data_2vectors/rgb.json")
    private Resource rgbResourceFile;

    private final TypeReference<List<Double>> embeddingType =   new TypeReference<List<Double>>() {};

    private final ObjectMapper mapper;

    private final VectorStore colorVectorStore;

    private final VectorStore vectorStore;

    public JsonDocLoader(VectorStore colorVectorStore, VectorStore vectorStore) {
        this.mapper = new ObjectMapper();
        this.colorVectorStore = colorVectorStore;
        this.vectorStore = vectorStore;
    }

    public void run() throws Exception {
        JsonNode rgbDocs = mapper.readTree(rgbResourceFile.getFile());
        List<Document> docs = new ArrayList<>();
        List<Document> colorDocs = new ArrayList<>();
        for (int i = 0; i < rgbDocs.size(); i++) {
            var node = rgbDocs.get(i);
            Map<String, Object> metadata = Map.of("brightness", node.get("brightness").asDouble(),
                "wheel_pos", node.get("wheel_pos").asText(),
                "verbs", node.get("verbs"),
                "embedding_model", node.get("embedding_model"),
                "description", node.get("description"),
                "color", node.get("color").asText()
            );

            var doc = new Document(node.get("id").asText(),
                node.get("color").asText(),
                metadata
            );
            doc.setEmbedding(mapper.convertValue(node.get("colorvect_l2"), embeddingType));
            colorDocs.add(doc);

            var doc2 = new Document(node.get("id").asText(),
                node.get("description").asText(),
                metadata
            );
            doc2.setEmbedding(mapper.convertValue(node.get("embedding_vector_dot"), embeddingType));
            docs.add(doc2);
        }
        colorVectorStore.add(colorDocs);
        vectorStore.add(docs);
    }
}
