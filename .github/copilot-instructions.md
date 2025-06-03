<rules>
    I have programming experience but am less familiar with production development practices and machine learning. I am learning to use GitHub Copilot to assist with coding tasks.
    Thus when the tasks are more focused on these areas, please provide detailed instructions and explanations so I understand the code and can learn from it.

    Use all terminal commands in a windows powershell terminal, making sure to include ; between statements.
    
    <memory>
    When you create a new file, update [file](../docs/memory.md) with a brief description of the file's purpose and any relevant details.
    </memory>

    <commit>
    WHEN file changes are COMPLETE:
    - Stage your changes with git add .
    - Commit them with an short generated message describing the changes starting with the step number, e.g. STEP #1 - <short description of changes>
    - Do this within a single terminal command using &&

    ONLY do this if you create or edit a file during the turn.
    </commit>
    
    <context>
    If you lack context on how to solve the user's request:
    
    FIRST, use #tool:resolve-library-id from Context7 to find the referenced library.

    NEXT, use #tool:get-library-docs from Context7 to get the library's documentation to assist in the user's request.
    </context>
</rules>