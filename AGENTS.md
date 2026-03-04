Project Name: Nayhein-8.8B-MDRM
Directory: mmdrllm./

## Gross Overview

You will create a Transformer based Auto-Regressive Reasoning Model using Qwen/Qwen3.5-0.8B-Base as Base Model finetuning to think in <think></think> tags. After that it should influence the Diffusion Model to output text following the output tags using GSAI-ML/LLaDA-V as the Diffusion Large Language Model.

## Naming Convention

Why call it that? Nayhein is going to be an Organisation i create once we are done with the project. 8B stands for the amount of parameters after reasoning .8B stands for the 800 M params used for reasoning in that pass

## Base Model information

- Qwen3.5 = Auto Regressive Text-Image-To-Text Model
- Finetuneable via TRL or Unsloth
- LLaDa-V = Diffusion  Text-Image-To-Text Model
  - Is finetunable via the github (mentioned in the readme <https://github.com/ML-GSAI/LLaDA-V>)

## Finetuning Qwen3.5

You will create a finetuning script for the 0.8B Base model. It must reason within Header tags

Example:

```md
**Designing a scalable API**

The user wants to design a scalable REST API using Node.js and Express that includes multi-tenant authentication and rate limiting. This involves outlining the architecture, including stateless design, JWT for authentication, and required middleware. I need to consider a database schema that includes a tenants table and address routing and error handling effectively. For multi-tenancy, I’ll evaluate options like database-per-tenant or schema-per-tenant. I'll also include a plan for rate limiting and relevant libraries. A concise overview that mixes structure with some sample code should work well!

**Structuring the API design**

I’m thinking that I need more structure in the API design by including headings such as Requirements & Assumptions, High-level Architecture, Multi-tenant Authentication, Rate Limiting, Example Code Skeleton, and Security & Scalability Considerations. I don’t need to dive deeply into every detail but should include some code for middleware that handles authentication and rate limiting using JWT, tenantId, and roles. I’ll illustrate how to retrieve tenant configurations. Let's create a clear layout to guide the design effectively!

**Detailing the architecture flow**

I’m mapping out the authentication flow that includes a POST request to /auth/login, which verifies the user within a tenant and issues a JWT with the subject, tenantId, and roles. The middleware should verify the JWT by decoding it and checking that the tenantId from the token matches the route parameter or header to deny cross-tenant access. 

Next, for rate limiting, I’ll use express-rate-limit with Redis to manage the limits for each tenant and user, plus factor in a folder structure for organization. Additionally, I'll ensure scalability with stateless design and cloud load balancers!

**Choosing the right setup**

It looks like the user hasn't specified the audience, but I’m guessing it's at a developer level. I’ll decide to code in JavaScript and opt for the modern syntax with ES modules since Node 18+ supports it. But I’m aware both CommonJS and ES modules are acceptable. To reduce confusion, I’ll include comments in the code.  

First, I’ll prepare by installing the necessary packages using npm. For the multi-tenant setup, I plan to use a path-based approach like `/api/:tenantId/...` to keep it clear, with the tenantValidation middleware ensuring users belong to that tenant.

**Crafting a scalable REST API**

I’m outlining the implementation of a rate limiter using express-rate-limit alongside Redis for multi-tenant authentication. First, I'll set up the rate limiter with a configuration for each tenant’s request limits. The JWT payload will include the user's ID, tenant ID, and role for proper authentication. I'll incorporate error handling and suggest using HTTPS in production.

Next, I'll develop a structured architecture with high-level design points, a description of core multi-tenancy concepts, and example implementations for key files like app.js and middleware. I’ll emphasize scalability best practices without overloading the explanation!
```

Must also have --generate-synthetic-data arg with lmstudio and ollama support and cli model choosing using a framework that generates  synthetic user prompts and the reasoning  response

then the Hybrid Auto-Regressive-Reasoning to Diffusion Output should be a normal process and the model should be loadable using AutoModelVision.
