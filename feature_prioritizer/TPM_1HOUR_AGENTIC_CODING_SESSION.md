# TPM 1-Hour Agentic Programming Learning Session

## 🎯 **Session Overview for Technical Program Managers**

This 1-hour session is designed for TPMs with limited coding experience to understand and modify agentic programming concepts. You'll learn by making practical changes to the Feature Prioritization Assistant.

## 📋 **Session Structure**

### **10 minutes**: Understanding Agentic Architecture
### **35 minutes**: Hands-on Coding (3 exercises)
### **15 minutes**: Testing and Validation

---

## 🧠 **Part 1: Quick Agentic Concepts (10 minutes)**

### What Makes Code "Agentic"?

**Traditional Code:**
```python
if "payment" in description:
    return 0.8  # Fixed rule
```

**Agentic Code:**
```python
def payment_analysis_agent(description):
    """I am an agent that analyzes payment features"""
    # Agent reasons about the specific context
    return llm_analysis(description, my_expertise="payment_systems")
```

**Key Differences:**
- ✅ **Agents have identity** and specific roles
- ✅ **Agents make decisions** based on context  
- ✅ **Agents can reason** and explain their decisions
- ✅ **Agents collaborate** by passing information

---

## 🛠 **Part 2: Hands-On Coding Exercises (35 minutes)**

### **Exercise 1: Customize Business Keywords (15 minutes)**
*Perfect for TPMs - No complex programming required*

**What You'll Learn:**
- How agents use business knowledge
- How to encode domain expertise
- Impact of keyword mappings on decisions

**Your Task:** Add your company's specific business keywords

**File to Edit:** `config.py` (lines 42-85)

**Current Code:**
```python
keyword_mappings: Dict[str, Dict[str, float]] = Field(
    default_factory=lambda: {
        # High revenue keywords
        "pricing": {"revenue": 0.9, "engineering": 0.4},
        "payment": {"revenue": 0.9, "engineering": 0.6},
        "subscription": {"revenue": 0.8, "engineering": 0.6},
        
        # Add your keywords here...
    }
)
```

**Your Mission:** Add 5 keywords specific to your business domain

**Example for E-commerce:**
```python
# Add these to the keyword_mappings dictionary:
"cart-abandonment": {"revenue": 0.8, "reach": 0.7},
"personalization": {"revenue": 0.7, "complexity": 0.6},
"inventory": {"risk_reduction": 0.8, "engineering": 0.5},
"recommendations": {"revenue": 0.6, "complexity": 0.7},
"wishlist": {"reach": 0.6, "engineering": 0.3},
```

**Validation Test:**
```bash
python run.py --json '[{"name": "Cart abandonment email", "description": "Send personalized cart-abandonment emails with inventory updates"}]' --verbose
```

**Expected Outcome:** The agent should recognize your keywords and adjust factors accordingly.

---

### **Exercise 2: Create a Custom Agent Prompt (15 minutes)**
*Learn how agents think and communicate*

**What You'll Learn:**
- How to give agents personality and expertise
- How agent identity affects decision-making
- How to constrain agent behavior

**Your Task:** Create a domain-specific agent prompt

**File to Edit:** `nodes.py` (lines 98-125)

**Current Agent:**
```python
prompt = f"""You are an agent for feature analysis and factor assessment. Your role is to analyze product features and estimate normalized business factors."""
```

**Your Mission:** Create a specialized agent for your domain

**Example for FinTech:**
```python
prompt = f"""You are an agent for financial technology feature analysis. Your expertise includes:
- Regulatory compliance requirements (SOX, PCI-DSS, GDPR)
- Financial risk assessment and fraud prevention
- Banking integration complexity and security protocols
- Customer trust and adoption patterns in financial services

As a FinTech agent, you prioritize:
- Security and compliance as critical risk factors
- Customer trust and regulatory approval for reach assessment
- Integration complexity with banking systems for effort estimation

Analyze this financial feature: {feature_desc}

Apply your FinTech expertise to estimate factors..."""
```

**Alternative for Healthcare:**
```python
prompt = f"""You are an agent for healthcare technology feature analysis. Your expertise includes:
- HIPAA compliance and patient data protection
- Clinical workflow integration and provider adoption
- Patient safety and care quality impact assessment
- Healthcare system interoperability challenges

As a Healthcare agent, you prioritize:
- Patient safety and regulatory compliance as top risk factors
- Provider workflow disruption for effort assessment
- Patient care quality improvement for impact evaluation

Analyze this healthcare feature: {feature_desc}

Apply your healthcare expertise to estimate factors..."""
```

**Validation Test:**
```bash
python run.py --json '[{"name": "HIPAA-compliant messaging", "description": "Secure patient-provider messaging with audit trail"}]' --llm --verbose
```

**Expected Outcome:** Your specialized agent should show domain expertise in its analysis.

---

### **Exercise 3: Modify Agent Collaboration (5 minutes)**
*Understand how agents work together*

**What You'll Learn:**
- How agents pass information between stages
- How agent decisions build on each other
- How to add agent reasoning transparency

**Your Task:** Add collaborative reasoning between agents

**File to Edit:** `nodes.py` (lines 220-230)

**Current Code:**
```python
if llm_analysis:
    # Override defaults with LLM analysis where available
    for factor in ['reach', 'revenue', 'risk_reduction', 'engineering', 'dependency', 'complexity']:
        if factor in llm_analysis and llm_analysis[factor] is not None:
            defaults[factor] = max(0.0, min(1.0, llm_analysis[factor]))
    
    # Add LLM analysis notes
    if 'notes' in llm_analysis and llm_analysis['notes']:
        final_notes.extend([f"LLM: {note}" for note in llm_analysis['notes']])
```

**Your Mission:** Add agent collaboration notes

**Enhanced Code:**
```python
if llm_analysis:
    # Override defaults with LLM analysis where available
    for factor in ['reach', 'revenue', 'risk_reduction', 'engineering', 'dependency', 'complexity']:
        if factor in llm_analysis and llm_analysis[factor] is not None:
            old_value = defaults[factor]
            new_value = max(0.0, min(1.0, llm_analysis[factor]))
            defaults[factor] = new_value
            
            # Add collaboration transparency
            if abs(new_value - old_value) > 0.2:  # Significant change
                final_notes.append(f"Agent Override: {factor} changed from {old_value:.2f} to {new_value:.2f}")
    
    # Add LLM analysis notes with agent attribution
    if 'notes' in llm_analysis and llm_analysis['notes']:
        final_notes.extend([f"Analysis Agent: {note}" for note in llm_analysis['notes']])
        final_notes.append(f"Agent Collaboration: {len(llm_analysis['notes'])} insights provided")
```

**Validation Test:**
```bash
python run.py --file samples/features.json --llm --verbose | grep -i "agent"
```

**Expected Outcome:** See agent collaboration and decision transparency in the output.

---

## 🧪 **Part 3: Testing and Validation (15 minutes)**

### **Test Your Agentic Modifications**

**1. Unit Test Your Keywords (5 minutes)**
```bash
# Test your new business keywords
python run.py --json '[{"name": "Test Feature", "description": "A feature containing your-new-keyword"}]' --verbose

# Verify the agent recognizes your keywords
# Look for your keyword in the analysis notes
```

**2. Test Your Custom Agent (5 minutes)**
```bash
# Test with domain-specific features
python run.py --json '[{"name": "Domain Feature", "description": "Feature relevant to your domain"}]' --llm --verbose

# Check if your agent shows domain expertise
# Look for specialized analysis in the output
```

**3. Validate Agent Collaboration (5 minutes)**
```bash
# Test the enhanced collaboration
python run.py --file samples/features.json --llm --verbose --auto-save

# Check the saved results for:
# - Agent override messages
# - Collaboration transparency notes  
# - Decision explanations
```

---

## 🎓 **Learning Outcomes**

By the end of this session, you'll understand:

### **1. Agentic Architecture Concepts**
- ✅ Agents have specific roles and expertise
- ✅ Agents make context-aware decisions
- ✅ Agents collaborate and build on each other's work
- ✅ Agents can explain their reasoning

### **2. Practical Agentic Programming**
- ✅ How to encode business knowledge in agents
- ✅ How to create specialized agent personalities
- ✅ How to add transparency to agent decisions
- ✅ How to test and validate agent behavior

### **3. Business Impact**
- ✅ How domain expertise improves AI decision-making
- ✅ How agent specialization reduces hallucination
- ✅ How agent collaboration increases transparency
- ✅ How to scale AI expertise across your organization

---

## 📈 **Next Steps for TPMs**

### **Immediate Actions (This Week)**
1. **Customize Keywords**: Add your domain-specific business terms
2. **Test with Real Data**: Use your actual feature backlogs
3. **Share with Team**: Demonstrate the customized agent behavior

### **Short-term Expansion (Next Month)**
1. **Create Agent Personas**: Develop agents for different product areas
2. **Add Business Rules**: Incorporate your specific prioritization criteria
3. **Integration Planning**: Connect with your existing tools and processes

### **Long-term Vision (Next Quarter)**
1. **Multi-Agent Systems**: Agents for different stakeholder perspectives
2. **Learning Agents**: Systems that improve based on your decisions
3. **Organization-wide Deployment**: Scale intelligent prioritization across teams

---

## 🔧 **Troubleshooting Guide**

### **Common Issues:**

**1. Keywords Not Working**
```python
# Check syntax - keywords are case-sensitive
"machine-learning": {"complexity": 0.8}  # ✅ Correct
"Machine Learning": {"complexity": 0.8}  # ❌ Won't match
```

**2. Agent Prompt Too Long**
```python
# Keep prompts focused and concise
# Aim for 500-800 characters for best performance
```

**3. No LLM Response**
```bash
# Check your API key
echo $OPENAI_API_KEY

# Verify .env file exists
cat /Users/n0m08hp/Agents/.env
```

---

## 💡 **Pro Tips for TPMs**

### **1. Start Small**
- Begin with 3-5 keywords specific to your domain
- Test each change before adding more
- Build confidence with simple modifications first

### **2. Think Like a Product Manager**
- What expertise would you want in a prioritization consultant?
- What business knowledge is unique to your organization?
- What assumptions do your team members make that an agent should know?

### **3. Validate Business Logic**
- Test with features you know well
- Compare agent decisions to your intuition
- Adjust keywords based on business outcomes

### **4. Document Your Customizations**
- Keep track of keywords and their rationale
- Document agent personalities and their expertise areas
- Share learnings with other TPMs in your organization

---

## 🚀 **Ready to Start?**

**Clone the repository and begin your agentic programming journey:**

```bash
git clone https://gecgithub01.walmart.com/n0m08hp/Agents.git
cd Agents/feature_prioritizer
pip install -r requirements.txt
```

**Your first agentic modification awaits!** 🤖✨