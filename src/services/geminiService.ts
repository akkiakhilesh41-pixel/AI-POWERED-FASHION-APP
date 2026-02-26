import { GoogleGenAI, Type } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

export interface FashionAnalysis {
  color: string;
  pattern: string;
  style: string;
  category: string;
  description: string;
}

export interface Recommendation {
  name: string;
  category: string;
  reason: string;
  platform: string;
  priceRange: string;
  imageUrl: string;
  purchaseUrl: string;
  matchScore: number;
}

export const analyzeFashionItem = async (base64Image: string): Promise<{ analysis: FashionAnalysis, recommendations: Recommendation[] }> => {
  const model = "gemini-3-flash-preview"; 
  
  const prompt = `Analyze this clothing item and provide:
  1. Detailed attributes (color, pattern, style, category).
  2. 3-4 matching outfit recommendations (e.g., if it's a shirt, suggest pants, shoes, and an accessory).
  
  CRITICAL: For each recommendation, you MUST use Google Search to find REAL, VALID, and CURRENT purchase links from Myntra, Amazon.in, Ajio, or Flipkart. 
  The 'purchaseUrl' field MUST be a direct link to a search result or product page on these platforms for the suggested item.
  
  Provide a realistic item name, the platform, a price range in INR, and a match score (0-100).
  Also provide a reason why it matches.`;

  const response = await ai.models.generateContent({
    model,
    contents: [
      {
        parts: [
          { text: prompt },
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: base64Image.split(",")[1] || base64Image,
            },
          },
        ],
      },
    ],
    config: {
      tools: [{ googleSearch: {} }],
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          analysis: {
            type: Type.OBJECT,
            properties: {
              color: { type: Type.STRING },
              pattern: { type: Type.STRING },
              style: { type: Type.STRING },
              category: { type: Type.STRING },
              description: { type: Type.STRING },
            },
            required: ["color", "pattern", "style", "category", "description"],
          },
          recommendations: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                name: { type: Type.STRING },
                category: { type: Type.STRING },
                reason: { type: Type.STRING },
                platform: { type: Type.STRING },
                priceRange: { type: Type.STRING },
                imageUrl: { type: Type.STRING },
                purchaseUrl: { type: Type.STRING },
                matchScore: { type: Type.NUMBER },
              },
              required: ["name", "category", "reason", "platform", "priceRange", "imageUrl", "purchaseUrl", "matchScore"],
            },
          },
        },
        required: ["analysis", "recommendations"],
      },
    },
  });

  return JSON.parse(response.text || "{}");
};

export const getStylistResponse = async (message: string, history: { role: string, content: string }[]) => {
  const model = "gemini-3-flash-preview";
  
  // Map history to the format expected by generateContent
  const contents = history.map(h => ({
    role: h.role === 'user' ? 'user' : 'model',
    parts: [{ text: h.content }]
  }));
  contents.push({ role: 'user', parts: [{ text: message }] });

  const response = await ai.models.generateContent({
    model,
    contents,
    config: {
      systemInstruction: `You are LUMIÈRE, a high-end luxury fashion stylist. 
      Your tone is warm, sophisticated, and encouraging. 
      When a user asks for styling advice:
      1. Provide a 'friendlyResponse' using Markdown. Use bold text for key items and bullet points for clarity. 
         Structure it like: A warm greeting, the 'The Vision' (overall vibe), 'The Components' (Top, Bottom, Shoes, Accessories), 'The Hairstyle' (suggested hair look), and a 'Stylist Tip'.
      2. Provide a 'visualPrompt' which is a concise, descriptive prompt for an image generation model of the COMPLETE OUTFIT.
      3. Provide a 'hairVisualPrompt' which is a concise, descriptive prompt for an image generation model focusing ONLY on the HAIRSTYLE (close-up).`,
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          friendlyResponse: { type: Type.STRING },
          visualPrompt: { type: Type.STRING },
          hairVisualPrompt: { type: Type.STRING }
        },
        required: ["friendlyResponse", "visualPrompt", "hairVisualPrompt"]
      }
    }
  });

  const data = JSON.parse(response.text || "{}");
  const friendlyText = data.friendlyResponse;
  const visualPrompt = data.visualPrompt;
  const hairVisualPrompt = data.hairVisualPrompt;

  // Generate Outfit Image
  const outfitImageResponse = await ai.models.generateContent({
    model: 'gemini-2.5-flash-image',
    contents: {
      parts: [{ text: `A high-end, professional fashion editorial photograph of a complete outfit: ${visualPrompt}. Minimalist luxury studio, 8k.` }],
    },
    config: { imageConfig: { aspectRatio: "3:4" } },
  });

  // Generate Hairstyle Image
  const hairImageResponse = await ai.models.generateContent({
    model: 'gemini-2.5-flash-image',
    contents: {
      parts: [{ text: `A high-end, professional beauty photography close-up of a hairstyle: ${hairVisualPrompt}. Minimalist luxury studio, soft lighting, 8k.` }],
    },
    config: { imageConfig: { aspectRatio: "1:1" } },
  });

  let imageUrl = "";
  for (const part of outfitImageResponse.candidates[0].content.parts) {
    if (part.inlineData) {
      imageUrl = `data:image/png;base64,${part.inlineData.data}`;
      break;
    }
  }

  let hairImageUrl = "";
  for (const part of hairImageResponse.candidates[0].content.parts) {
    if (part.inlineData) {
      hairImageUrl = `data:image/png;base64,${part.inlineData.data}`;
      break;
    }
  }

  return { text: friendlyText, imageUrl, hairImageUrl };
};
