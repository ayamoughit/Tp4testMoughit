package ma.emsi.moughit;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Scanner;

public class TestRoutage {

    // Configuration du logger pour voir les requêtes/responses du LM
    public static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {
        configureLogger();

        String llmKey = System.getenv("GEMINI_KEY");

        ChatModel chatModel = GoogleAiGeminiChatModel
                .builder()
                .apiKey(llmKey)
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .modelName("gemini-2.5-flash")
                .build();

        // --- Document Ingestion ---
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document docAI = FileSystemDocumentLoader.loadDocument(toPath("rag.pdf"), parser);
        Document docNonAI = FileSystemDocumentLoader.loadDocument(toPath("Chapitre-06__Pig-Latin.pdf"), parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segmentsAI = splitter.split(docAI);
        List<TextSegment> segmentsNonAI = splitter.split(docNonAI);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        List<Embedding> embeddingsAI = embeddingModel.embedAll(segmentsAI).content();
        List<Embedding> embeddingsNonAI = embeddingModel.embedAll(segmentsNonAI).content();

        EmbeddingStore<TextSegment> storeAI = new InMemoryEmbeddingStore<>();
        storeAI.addAll(embeddingsAI, segmentsAI);

        EmbeddingStore<TextSegment> storeNonAI = new InMemoryEmbeddingStore<>();
        storeNonAI.addAll(embeddingsNonAI, segmentsNonAI);

        // --- Content Retrievers ---
        ContentRetriever retrieverAI = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeAI)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        ContentRetriever retrieverNonAI = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeNonAI)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        // --- Query Router ---
        Map<ContentRetriever, String> routerMap = new HashMap<>();
        routerMap.put(retrieverAI, "This document contains information about Artificial Intelligence, RAG, and related concepts.");
        routerMap.put(retrieverNonAI, "This document contains information about general problems, heuristics, and non-AI topics.");

        QueryRouter queryRouter = new LanguageModelQueryRouter(chatModel, routerMap);

        // --- Retrieval Augmentor ---
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // --- Assistant ---
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // --- Chat loop ---
        System.out.println("Assistant is ready. Ask your questions (type 'exit' to quit).");
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("User: ");
            String question = scanner.nextLine();
            if ("exit".equalsIgnoreCase(question)) break;

            String response = assistant.chat(question);
            System.out.println("Assistant: " + response);
        }
        scanner.close();
    }

    private static Path toPath(String resourceName) {
        try {
            URL resourceUrl = TestRoutage.class.getClassLoader().getResource(resourceName);
            if (resourceUrl == null) throw new RuntimeException("Resource not found: " + resourceName);
            return Paths.get(resourceUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }
}