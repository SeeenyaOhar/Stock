package com.senyastr.stockweb.services.InquiryAnalyzer;

import com.senyastr.stockweb.models.MessageClass;
import org.springframework.boot.configurationprocessor.json.JSONException;
import org.springframework.boot.configurationprocessor.json.JSONObject;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class InquiryAnalyzerBERTService implements InquiryAnalyzerService {
    private final RestTemplate restTemplate = new RestTemplate();
    @Override
    public InquirySummary analyzeInquiry(String message) {
        String API_URL = "http://127.0.0.1:5000";
        String ANALYZE_ENDPOINT = "/class_message";
        try {
            JSONObject messageJsonObject = new JSONObject();
            messageJsonObject.put("message", message);
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<String> request = new HttpEntity<String>(messageJsonObject.toString(), headers);
            String result = restTemplate.postForObject(API_URL + ANALYZE_ENDPOINT, request, String.class);
            assert result != null;
            return new InquirySummary(Integer.parseInt(result));
        } catch (NumberFormatException e) {
            System.out.println("Number format exception.\nResponse body from Inquiry Analyzer is broken!\n" + e);
            return new InquirySummary(MessageClass.USER_INTERACTION_NEEDED.ordinal());
        } catch (JSONException jsonE) {
            System.out.println("Some problems with JSON in InquiryBERTService\n" + jsonE);
            return new InquirySummary(MessageClass.USER_INTERACTION_NEEDED.ordinal());
        }
    }
}
