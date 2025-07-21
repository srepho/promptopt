#!/usr/bin/env python3
"""Examples of using the flexible data generation system."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptopt.data import (
    FlexibleDataGenerator,
    TemplateBuilder,
    FieldType,
    FieldDefinition
)
from promptopt.utils import create_llm_client
import json


def example_ecommerce_reviews():
    """Generate e-commerce product review data."""
    print("=== E-commerce Review Generation ===\n")
    
    # Create generator
    generator = FlexibleDataGenerator()
    
    # Build custom template for product reviews
    template = (TemplateBuilder("product_reviews")
        .with_description("Generate realistic product reviews with ratings")
        .add_input_field("product_name", "text", description="Name of the product")
        .add_input_field("product_category", "enum:electronics,clothing,home,sports,books")
        .add_input_field("price", "number", min_value=9.99, max_value=999.99)
        .add_input_field("customer_id", "generator:uuid")
        .add_input_field("purchase_date", "date", format="%Y-%m-%d")
        .add_output_field("rating", "enum:1,2,3,4,5", 
                         weight={5: 0.4, 4: 0.3, 3: 0.15, 2: 0.1, 1: 0.05})
        .add_output_field("review_title", "text", description="Short review title")
        .add_output_field("review_text", "text", description="Detailed review")
        .add_output_field("verified_purchase", "enum:true,false", weight={"true": 0.8, "false": 0.2})
        .add_output_field("helpful_count", "number", min_value=0, max_value=100)
        .with_input_template("""Product: {product_name}
Category: {product_category}
Price: ${price}
Customer: {customer_id}
Purchase Date: {purchase_date}""")
        .with_output_template("""Rating: {rating}/5 stars
Title: {review_title}
Review: {review_text}
Verified Purchase: {verified_purchase}
Helpful: {helpful_count} people found this helpful""")
        .add_validation(lambda i, o: int(o["rating"]) >= 1 and int(o["rating"]) <= 5)
        .build()
    )
    
    # Register template
    generator.register_template(template)
    
    # Generate data
    dataset = generator.generate("product_reviews", count=3)
    
    # Display results
    for i, example in enumerate(dataset.examples[:3]):
        print(f"Example {i+1}:")
        print("INPUT:")
        print(example.input)
        print("\nOUTPUT:")
        print(example.output)
        print("-" * 50)


def example_financial_transactions():
    """Generate financial transaction data with complex rules."""
    print("\n=== Financial Transaction Generation ===\n")
    
    generator = FlexibleDataGenerator()
    
    # Register custom generators for financial data
    generator.register_generator("account_number", 
        lambda ctx, data: f"ACC-{ctx.get('bank_code', 'XX')}-" + 
                         ''.join([str(i) for i in range(10)])[:8])
    
    generator.register_generator("transaction_amount",
        lambda ctx, data: round(random.uniform(0.01, 10000.00), 2)
        if data.get("transaction_type") != "withdrawal"
        else round(random.uniform(20.00, 500.00), 2))
    
    # Create template with composite fields
    template = generator.create_template(
        name="financial_transactions",
        description="Bank transaction records with fraud detection",
        input_spec={
            "transaction_id": "generator:transaction_id",
            "timestamp": "generator:timestamp",
            "account": {
                "type": "composite",
                "sub_fields": {
                    "number": {"type": "function", "generator": "account_number"},
                    "type": {"type": "enum", "options": ["checking", "savings", "credit"]},
                    "balance": {"type": "number", "min": 0, "max": 100000}
                }
            },
            "transaction_type": "enum:deposit,withdrawal,transfer,payment",
            "amount": "generator:transaction_amount",
            "merchant": "generator:company",
            "location": {
                "type": "composite",
                "sub_fields": {
                    "city": {"type": "text"},
                    "country": {"type": "enum", "options": ["US", "CA", "UK", "AU"]},
                    "ip_address": {"type": "regex", "pattern": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"}
                }
            }
        },
        output_spec={
            "fraud_score": "number",
            "fraud_detected": "enum:true,false",
            "risk_factors": "json",
            "action_taken": "enum:approved,declined,review",
            "notification_sent": "enum:true,false"
        }
    )
    
    generator.register_template(template)
    
    # Generate with business context
    context = {"bank_code": "WF", "region": "North America"}
    dataset = generator.generate("financial_transactions", count=2, context=context)
    
    for i, example in enumerate(dataset.examples):
        print(f"\nTransaction {i+1}:")
        print("INPUT DATA:")
        print(json.dumps(example.metadata['input_data'], indent=2))
        print("\nOUTPUT DATA:")
        print(json.dumps(example.metadata['output_data'], indent=2))
        print("-" * 50)


def example_healthcare_records():
    """Generate healthcare/medical record data."""
    print("\n=== Healthcare Record Generation ===\n")
    
    # Initialize with LLM for better text generation (optional)
    llm_client = None  # create_llm_client("openai", "gpt-3.5-turbo") if API key available
    generator = FlexibleDataGenerator(llm_client)
    
    # Define custom validation rules
    def validate_vitals(input_data, output_data):
        # Ensure vital signs are within reasonable ranges
        vitals = output_data.get("vital_signs", {})
        bp = vitals.get("blood_pressure", "120/80").split("/")
        systolic = int(bp[0])
        return 90 <= systolic <= 180
    
    def validate_diagnosis_consistency(input_data, output_data):
        # Ensure diagnosis matches symptoms somewhat
        symptoms = input_data.get("symptoms", "").lower()
        diagnosis = output_data.get("diagnosis", "").lower()
        
        # Simple consistency check
        if "fever" in symptoms and "infection" not in diagnosis:
            return False
        return True
    
    # Build healthcare template
    template = (TemplateBuilder("patient_records")
        .with_description("Generate patient visit records with diagnosis")
        # Input fields - patient presentation
        .add_input_field("patient_id", "regex:[A-Z]{2}\\d{6}")
        .add_input_field("age", "number", min_value=1, max_value=100)
        .add_input_field("gender", "enum:M,F,Other")
        .add_input_field("visit_type", "enum:routine,urgent,emergency,follow-up")
        .add_input_field("symptoms", "text", description="Patient reported symptoms")
        .add_input_field("medical_history", "json", sub_fields={
            "conditions": {"type": "text"},
            "medications": {"type": "text"},
            "allergies": {"type": "text"}
        })
        # Output fields - medical assessment
        .add_output_field("vital_signs", "json", sub_fields={
            "blood_pressure": {"type": "text"},
            "heart_rate": {"type": "number", "min": 40, "max": 180},
            "temperature": {"type": "number", "min": 95.0, "max": 105.0},
            "oxygen_saturation": {"type": "number", "min": 85, "max": 100}
        })
        .add_output_field("diagnosis", "text", description="Primary diagnosis")
        .add_output_field("treatment_plan", "text", description="Recommended treatment")
        .add_output_field("prescriptions", "json")
        .add_output_field("follow_up_required", "enum:yes,no")
        .add_output_field("referral", "text")
        # Templates
        .with_input_template("""PATIENT VISIT RECORD
Patient ID: {patient_id}
Age: {age} | Gender: {gender}
Visit Type: {visit_type}

Chief Complaint: {symptoms}

Medical History:
- Conditions: {medical_history.conditions}
- Current Medications: {medical_history.medications}
- Allergies: {medical_history.allergies}""")
        .with_output_template("""ASSESSMENT AND PLAN

Vital Signs:
- BP: {vital_signs.blood_pressure}
- HR: {vital_signs.heart_rate} bpm
- Temp: {vital_signs.temperature}°F
- O2 Sat: {vital_signs.oxygen_saturation}%

Diagnosis: {diagnosis}

Treatment Plan: {treatment_plan}

Prescriptions: {prescriptions}

Follow-up Required: {follow_up_required}
Referral: {referral}""")
        # Add validations
        .add_validation(validate_vitals)
        .add_validation(validate_diagnosis_consistency)
        .build()
    )
    
    generator.register_template(template)
    
    # Generate records
    dataset = generator.generate("patient_records", count=2)
    
    for example in dataset.examples:
        print("\n" + "="*60)
        print(example.input)
        print("\n" + "-"*30 + "\n")
        print(example.output)


def example_custom_api_testing():
    """Generate custom API request/response pairs."""
    print("\n=== Custom API Testing Data ===\n")
    
    generator = FlexibleDataGenerator()
    
    # Register custom generators for API data
    generator.register_generator("http_method", 
        lambda ctx, data: random.choice(["GET", "POST", "PUT", "DELETE", "PATCH"]))
    
    generator.register_generator("status_code",
        lambda ctx, data: random.choice([200, 201, 400, 401, 404, 500])
        if random.random() > 0.8 else 200)
    
    generator.register_generator("response_time",
        lambda ctx, data: round(random.uniform(50, 2000), 2))
    
    # Build API testing template
    template = generator.create_template(
        name="api_tests",
        description="Generate API test cases with requests and responses",
        input_spec={
            "endpoint": "template:/api/v1/{resource}/{id}",
            "method": "generator:http_method",
            "headers": {
                "type": "json",
                "sub_fields": {
                    "Authorization": {"type": "template", "pattern": "Bearer {token}"},
                    "Content-Type": {"type": "enum", "options": ["application/json", "application/xml"]},
                    "X-Request-ID": {"type": "function", "generator": "uuid"}
                }
            },
            "query_params": "json",
            "body": "json"
        },
        output_spec={
            "status_code": "generator:status_code",
            "response_headers": {
                "type": "json",
                "sub_fields": {
                    "Content-Type": {"type": "text"},
                    "X-Response-Time": {"type": "function", "generator": "response_time"},
                    "X-Rate-Limit-Remaining": {"type": "number", "min": 0, "max": 1000}
                }
            },
            "response_body": "json",
            "response_time_ms": "generator:response_time",
            "cache_hit": "enum:true,false"
        }
    )
    
    generator.register_template(template)
    
    # Generate API test data
    context = {
        "resource": "users",
        "id": "12345",
        "token": "test-token-abc123"
    }
    
    dataset = generator.generate("api_tests", count=3, context=context)
    
    for i, example in enumerate(dataset.examples):
        print(f"\nAPI Test Case {i+1}:")
        print("REQUEST:")
        input_data = example.metadata['input_data']
        print(f"{input_data['method']} {input_data['endpoint']}")
        print(f"Headers: {json.dumps(input_data['headers'], indent=2)}")
        if input_data.get('body'):
            print(f"Body: {json.dumps(input_data['body'], indent=2)}")
        
        print("\nRESPONSE:")
        output_data = example.metadata['output_data']
        print(f"Status: {output_data['status_code']}")
        print(f"Response Time: {output_data['response_time_ms']}ms")
        print(f"Cache Hit: {output_data['cache_hit']}")
        print("-" * 50)


def example_with_llm_enhancement():
    """Example using LLM to generate more realistic content."""
    print("\n=== LLM-Enhanced Generation ===\n")
    
    # This example shows how LLM integration improves quality
    # Uncomment and add API key to test with real LLM
    
    # llm_client = create_llm_client("openai", "gpt-3.5-turbo")
    # generator = FlexibleDataGenerator(llm_client)
    
    generator = FlexibleDataGenerator()  # Without LLM for demo
    
    # Create a template for generating training data for a chatbot
    template = (TemplateBuilder("chatbot_training")
        .with_description("Generate conversational AI training data")
        .add_input_field("user_intent", "enum:greeting,question,complaint,request,feedback")
        .add_input_field("context", "text", description="Conversation context")
        .add_input_field("user_mood", "enum:neutral,happy,frustrated,confused")
        .add_output_field("bot_response", "text", description="Appropriate bot response")
        .add_output_field("suggested_actions", "json")
        .add_output_field("escalate_to_human", "enum:yes,no")
        .with_input_template("User ({user_mood}): [{user_intent}] {context}")
        .with_output_template("""Bot: {bot_response}
Actions: {suggested_actions}
Escalate: {escalate_to_human}""")
        .build()
    )
    
    generator.register_template(template)
    dataset = generator.generate("chatbot_training", count=3)
    
    print("Generated Chatbot Training Data:")
    for example in dataset.examples:
        print("\n" + example.input)
        print(example.output)
        print("-" * 40)


if __name__ == "__main__":
    import random
    random.seed(42)  # For reproducible examples
    
    # Run all examples
    example_ecommerce_reviews()
    example_financial_transactions()
    example_healthcare_records()
    example_custom_api_testing()
    example_with_llm_enhancement()
    
    print("\n✅ All examples completed!")
    print("\nThe flexible data generator allows you to:")
    print("- Define custom field types and generators")
    print("- Create complex nested data structures")
    print("- Add validation rules and constraints")
    print("- Use templates for formatting")
    print("- Integrate with LLMs for realistic text")
    print("- Build domain-specific data generators")