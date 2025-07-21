#!/usr/bin/env python3
"""Examples of using LLM-based data generation."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptopt.data.flexible_generator_v2 import (
    FlexibleDataGenerator,
    TemplateBuilder,
    LLMGenerationConfig
)
from promptopt.utils import create_llm_client
import json


def example_with_llm():
    """Generate data using LLM for realistic content."""
    print("=== LLM-Based Data Generation Example ===\n")
    
    # Create LLM client (requires API key)
    try:
        # Try OpenAI first
        llm_client = create_llm_client("openai", "gpt-3.5-turbo")
        print("✅ Using OpenAI for generation")
    except:
        try:
            # Try Anthropic
            llm_client = create_llm_client("anthropic", "claude-3-haiku-20240307")
            print("✅ Using Anthropic for generation")
        except:
            print("⚠️  No API keys found. Using fallback generation.")
            llm_client = None
    
    # Configure LLM generation
    config = LLMGenerationConfig(
        temperature=0.8,
        max_tokens=150,
        use_examples=True,
        use_context=True,
        fallback_to_simple=True
    )
    
    # Create generator with LLM
    generator = FlexibleDataGenerator(llm_client, config)
    
    # Example 1: Customer Reviews with LLM
    print("\n1. Customer Product Reviews")
    print("-" * 40)
    
    review_template = (TemplateBuilder("product_reviews")
        .with_description("Generate realistic product reviews")
        .add_input_field("product_name", "text", 
                        description="name of the product being reviewed")
        .add_input_field("product_category", "enum:electronics,clothing,books,home,sports",
                        description="category of product")
        .add_input_field("price_range", "enum:budget,mid-range,premium",
                        description="price tier of product")
        .add_input_field("verified_purchase", "enum:true,false",
                        weight={"true": 0.8, "false": 0.2})
        .add_output_field("rating", "enum:1,2,3,4,5",
                         weight={5: 0.3, 4: 0.4, 3: 0.15, 2: 0.1, 1: 0.05},
                         description="star rating based on experience")
        .add_output_field("title", "text",
                         description="short review title summarizing opinion")
        .add_output_field("review_text", "text",
                         description="detailed review with pros, cons, and personal experience")
        .add_output_field("helpful_aspects", "json",
                         description="list of helpful product features")
        .add_output_field("improvement_suggestions", "json",
                         description="list of suggested improvements")
        .with_input_template("""Product: {product_name}
Category: {product_category}
Price Range: {price_range}
Verified Purchase: {verified_purchase}""")
        .with_output_template("""Rating: {rating}/5 stars
Title: {title}

Review: {review_text}

Helpful Aspects: {helpful_aspects}
Improvements Needed: {improvement_suggestions}""")
        .build()
    )
    
    generator.register_template(review_template)
    
    # Generate with context
    context = {
        "store": "TechMart Online",
        "season": "holiday shopping",
        "promotion": "Black Friday"
    }
    
    # Generate specific product reviews
    products = [
        {"product_name": "UltraSound Pro Headphones", "product_category": "electronics", "price_range": "premium"},
        {"product_name": "ComfyFit Running Shoes", "product_category": "sports", "price_range": "mid-range"},
        {"product_name": "Smart Home Hub X", "product_category": "electronics", "price_range": "premium"}
    ]
    
    for i, product_context in enumerate(products):
        print(f"\nGenerating review for: {product_context['product_name']}")
        
        # Merge contexts
        full_context = {**context, **product_context}
        
        # Generate one example with this context
        dataset = generator.generate("product_reviews", count=1, context=full_context)
        
        example = dataset.examples[0]
        print("\nGenerated Review:")
        print(example.output)
        print("-" * 40)
    
    # Example 2: Support Tickets with Context
    print("\n\n2. Context-Aware Support Tickets")
    print("-" * 40)
    
    ticket_template = (TemplateBuilder("support_tickets")
        .with_description("Generate realistic support tickets")
        .add_input_field("customer_tier", "enum:free,basic,pro,enterprise",
                        description="customer subscription level")
        .add_input_field("issue_category", "enum:billing,technical,feature_request,bug_report",
                        description="type of support issue")
        .add_input_field("urgency", "enum:low,medium,high,critical",
                        description="urgency level of the issue")
        .add_input_field("product_area", "text",
                        description="specific product area affected")
        .add_output_field("ticket_title", "text",
                         description="concise ticket title")
        .add_output_field("issue_description", "text",
                         description="detailed description of the issue with steps to reproduce")
        .add_output_field("expected_behavior", "text",
                         description="what the customer expected to happen")
        .add_output_field("actual_behavior", "text",
                         description="what actually happened")
        .add_output_field("business_impact", "text",
                         description="how this affects the customer's business")
        .add_output_field("suggested_priority", "enum:P0,P1,P2,P3",
                         description="suggested priority based on impact")
        .build()
    )
    
    generator.register_template(ticket_template)
    
    # Generate tickets with business context
    business_context = {
        "company": "DataFlow Analytics",
        "industry": "fintech",
        "peak_season": True,
        "sla_tier": "platinum"
    }
    
    tickets = generator.generate("support_tickets", count=2, context=business_context)
    
    for i, ticket in enumerate(tickets.examples):
        print(f"\nTicket #{i+1}:")
        data = ticket.metadata['output_data']
        print(f"Title: {data.get('ticket_title', 'N/A')}")
        print(f"Priority: {data.get('suggested_priority', 'N/A')}")
        print(f"\nIssue: {data.get('issue_description', 'N/A')}")
        print(f"\nBusiness Impact: {data.get('business_impact', 'N/A')}")
        print("-" * 40)
    
    # Example 3: Industry-Specific Content
    print("\n\n3. Industry-Specific Email Generation")
    print("-" * 40)
    
    email_template = (TemplateBuilder("business_emails")
        .with_description("Generate industry-specific business emails")
        .add_input_field("sender_role", "text",
                        description="role of the email sender")
        .add_input_field("recipient_role", "text",
                        description="role of the recipient")
        .add_input_field("email_purpose", "enum:proposal,follow-up,announcement,request,update",
                        description="primary purpose of the email")
        .add_input_field("formality_level", "enum:casual,professional,formal",
                        description="required formality level")
        .add_output_field("subject_line", "text",
                         description="compelling email subject line",
                         constraints=["max 60 characters", "action-oriented"])
        .add_output_field("greeting", "text",
                         description="appropriate greeting based on formality")
        .add_output_field("body_paragraphs", "json",
                         description="list of paragraph texts forming the email body")
        .add_output_field("call_to_action", "text",
                         description="clear next steps or action items")
        .add_output_field("closing", "text",
                         description="appropriate email closing")
        .build()
    )
    
    generator.register_template(email_template)
    
    # Generate for different industries
    industries = [
        {
            "industry": "healthcare",
            "compliance": "HIPAA",
            "sender_role": "Medical Director",
            "recipient_role": "Department Heads",
            "email_purpose": "announcement",
            "formality_level": "professional"
        },
        {
            "industry": "finance",
            "compliance": "SOX",
            "sender_role": "CFO",
            "recipient_role": "Board of Directors",
            "email_purpose": "update",
            "formality_level": "formal"
        }
    ]
    
    for industry_context in industries:
        emails = generator.generate("business_emails", count=1, context=industry_context)
        email_data = emails.examples[0].metadata['output_data']
        
        print(f"\n{industry_context['industry'].title()} Email:")
        print(f"From: {industry_context['sender_role']}")
        print(f"To: {industry_context['recipient_role']}")
        print(f"Subject: {email_data.get('subject_line', 'N/A')}")
        print(f"\n{email_data.get('greeting', 'N/A')}")
        
        paragraphs = email_data.get('body_paragraphs', [])
        if isinstance(paragraphs, list):
            for para in paragraphs:
                print(f"\n{para}")
        
        print(f"\n{email_data.get('call_to_action', 'N/A')}")
        print(f"\n{email_data.get('closing', 'N/A')}")
        print("-" * 40)
    
    # Show generation cost
    if generator.get_generation_cost() > 0:
        print(f"\n💰 Total generation cost: ${generator.get_generation_cost():.4f}")


def example_without_llm():
    """Show fallback behavior without LLM."""
    print("\n\n=== Fallback Generation (No LLM) ===\n")
    
    # Create generator without LLM
    generator = FlexibleDataGenerator(llm_client=None)
    
    # Same template as before
    template = (TemplateBuilder("simple_data")
        .with_description("Simple data generation")
        .add_input_field("name", "text", description="person's name")
        .add_input_field("age", "number", min_value=18, max_value=80)
        .add_input_field("department", "enum:sales,engineering,marketing,support")
        .add_output_field("employee_id", "regex:[A-Z]{3}\\d{5}")
        .add_output_field("email", "generator:email")
        .add_output_field("start_date", "date")
        .build()
    )
    
    generator.register_template(template)
    dataset = generator.generate("simple_data", count=3)
    
    print("Generated data without LLM:")
    for i, example in enumerate(dataset.examples):
        print(f"\nExample {i+1}:")
        print(f"Input: {example.metadata['input_data']}")
        print(f"Output: {example.metadata['output_data']}")


def example_mixed_generation():
    """Example mixing LLM and non-LLM fields."""
    print("\n\n=== Mixed LLM/Non-LLM Generation ===\n")
    
    # Try to get LLM client
    try:
        llm_client = create_llm_client("openai", "gpt-3.5-turbo")
    except:
        llm_client = None
    
    config = LLMGenerationConfig(
        temperature=0.7,
        max_tokens=100,
        fallback_to_simple=True
    )
    
    generator = FlexibleDataGenerator(llm_client, config)
    
    # Template with mixed field types
    template = (TemplateBuilder("user_profiles")
        .with_description("User profile generation")
        # Non-LLM fields
        .add_input_field("user_id", "generator:uuid", use_llm=False)
        .add_input_field("created_date", "date", use_llm=False)
        .add_input_field("account_type", "enum:free,premium,enterprise", use_llm=False)
        # LLM fields
        .add_input_field("interests", "text", 
                        description="user's interests and hobbies", use_llm=True)
        .add_output_field("bio", "text",
                         description="engaging user bio based on interests", use_llm=True)
        .add_output_field("recommended_features", "json",
                         description="list of recommended features based on account type and interests",
                         use_llm=True)
        # Non-LLM fields
        .add_output_field("profile_completeness", "number", 
                         min_value=60, max_value=100, use_llm=False)
        .build()
    )
    
    generator.register_template(template)
    
    # Generate profiles
    profiles = generator.generate("user_profiles", count=2)
    
    for i, profile in enumerate(profiles.examples):
        print(f"\nUser Profile {i+1}:")
        input_data = profile.metadata['input_data']
        output_data = profile.metadata['output_data']
        
        print(f"User ID: {input_data.get('user_id', 'N/A')}")
        print(f"Account: {input_data.get('account_type', 'N/A')}")
        print(f"Interests: {input_data.get('interests', 'N/A')}")
        print(f"\nBio: {output_data.get('bio', 'N/A')}")
        print(f"Recommended Features: {output_data.get('recommended_features', 'N/A')}")
        print(f"Profile Completeness: {output_data.get('profile_completeness', 'N/A')}%")


if __name__ == "__main__":
    print("🚀 LLM-Enhanced Data Generation Examples\n")
    print("Note: Set OPENAI_API_KEY or ANTHROPIC_API_KEY for best results\n")
    
    # Run examples
    example_with_llm()
    example_without_llm()
    example_mixed_generation()
    
    print("\n✅ Examples completed!")
    print("\nKey Features Demonstrated:")
    print("- LLM-based generation for realistic content")
    print("- Context-aware generation")
    print("- Industry-specific content")
    print("- Fallback to simple generation without LLM")
    print("- Mixed LLM/non-LLM field generation")
    print("- Cost tracking for LLM usage")