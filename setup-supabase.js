#!/usr/bin/env node

/**
 * Setup script to create the necessary tables in Supabase for agent memory
 * 
 * Usage:
 *   npm install @supabase/supabase-js dotenv
 *   node setup-supabase.js
 * 
 * Environment Variables:
 *   SUPABASE_URL - The URL of your Supabase project
 *   SUPABASE_KEY - The service role key of your Supabase project
 */

const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');
require('dotenv').config();

// Configuration
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('Error: SUPABASE_URL and SUPABASE_KEY must be provided in .env file');
  console.error('Create a .env file with:');
  console.error('SUPABASE_URL=your-supabase-url');
  console.error('SUPABASE_KEY=your-supabase-service-key');
  process.exit(1);
}

// Initialize Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Read the SQL schema file
const schemaPath = path.join(__dirname, 'schema.sql');
let sqlScript;

try {
  sqlScript = fs.readFileSync(schemaPath, 'utf8');
} catch (error) {
  console.error(`Error reading schema file: ${error.message}`);
  process.exit(1);
}

// Split the SQL script into individual statements
const statements = sqlScript
  .replace(/\/\*[\s\S]*?\*\/|--.*$/gm, '') // Remove comments
  .split(';')
  .map(stmt => stmt.trim())
  .filter(stmt => stmt.length > 0);

async function executeStatements() {
  console.log('Setting up Supabase schema for agent memory...');
  console.log(`Found ${statements.length} SQL statements to execute.`);

  // Execute each statement
  for (let i = 0; i < statements.length; i++) {
    const statement = statements[i];
    const firstLine = statement.split('\n')[0].trim();
    
    try {
      console.log(`Executing statement ${i + 1}/${statements.length}: ${firstLine.slice(0, 50)}...`);
      const { error } = await supabase.rpc('run_sql_query', { query: statement });
      
      if (error) {
        console.error(`Error executing statement ${i + 1}: ${error.message}`);
        console.log('The statement was:');
        console.log(statement);
      } else {
        console.log(`Statement ${i + 1} executed successfully.`);
      }
    } catch (error) {
      console.error(`Exception executing statement ${i + 1}: ${error.message}`);
    }
  }

  console.log('Schema setup complete!');
}

// Set up basic tables for agent data
async function setupBasicTables() {
  console.log('Setting up basic tables...');

  // Create a custom function to execute SQL
  const { error: funcError } = await supabase.rpc(
    'create_sql_runner_function',
    {
      sql: `
        CREATE OR REPLACE FUNCTION run_sql_query(query text)
        RETURNS void
        LANGUAGE plpgsql
        SECURITY DEFINER
        AS $$
        BEGIN
          EXECUTE query;
        END;
        $$;
      `
    }
  );

  if (funcError) {
    console.error('Error creating SQL runner function:');
    console.error(funcError);
    
    // Try to create it directly
    console.log('Trying alternate method...');
    
    const { error } = await supabase.rpc('execute_sql', {
      sql_query: `
        CREATE OR REPLACE FUNCTION run_sql_query(query text)
        RETURNS void
        LANGUAGE plpgsql
        SECURITY DEFINER
        AS $$
        BEGIN
          EXECUTE query;
        END;
        $$;
      `
    });
    
    if (error) {
      console.error('Error creating SQL runner function (alternate method):');
      console.error(error);
      console.log('');
      console.log('You may need to create this function manually in the SQL editor:');
      console.log(`
        CREATE OR REPLACE FUNCTION run_sql_query(query text)
        RETURNS void
        LANGUAGE plpgsql
        SECURITY DEFINER
        AS $$
        BEGIN
          EXECUTE query;
        END;
        $$;
      `);
      console.log('Then run this script again.');
      process.exit(1);
    }
  }

  console.log('SQL runner function created.');
}

// Main function
async function main() {
  console.log('Setting up Supabase for XII-OS Agent Memory...');
  
  try {
    // First set up the SQL runner function
    await setupBasicTables();
    
    // Then execute all the statements in the schema file
    await executeStatements();
    
    console.log('\nSetup completed successfully!');
    console.log('\nYou can now use the Supabase memory store in your agents:');
    console.log('\n  from utils.supabase_memory import SupabaseMemoryStore');
    console.log('  memory = SupbabaseMemoryStore("agent_id")');
    console.log('\nMake sure to set SUPABASE_URL and SUPABASE_KEY in your environment.');
  } catch (error) {
    console.error('Error during setup:', error);
    process.exit(1);
  }
}

main(); 