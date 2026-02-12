import React from 'react';
import Editor from '@monaco-editor/react';
import { useCodeStore } from '../../store/codeStore';
import { syncCoordinator } from '../../services/sync/SyncCoordinator';

export default function PyTorchEditor() {
    const { code, updateFromUser } = useCodeStore();

    const handleEditorChange = (value) => {
        updateFromUser(value || '');
        syncCoordinator.onCodeChange(value || '');
    };

    return (
        <Editor
            height="100%"
            defaultLanguage="python"
            theme="vs-dark"
            value={code}
            onChange={handleEditorChange}
            options={{
                minimap: { enabled: true },
                fontSize: 14,
                fontFamily: 'Fira Code, Monaco, Courier New, monospace',
                lineNumbers: 'on',
                rulers: [80],
                wordWrap: 'on',
                automaticLayout: true,
            }}
        />
    );
}
