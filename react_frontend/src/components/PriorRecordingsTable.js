import { useContext, useState, useEffect } from 'react';
import axios from 'axios';
import { Table } from 'react-bootstrap';
import { AcquisitionState } from "../AcquistionApi";

const PriorRecordingsTable = ({ api }) => {

    const { priorRecordings } = useContext(AcquisitionState);

    return (
        <Table id="prior_recordings_table" striped bordered hover>
            <thead>
                <tr>
                    <th>Participant</th>
                    <th>Filename</th>
                    <th>Comment</th>
                </tr>
            </thead>
            <tbody>
                {priorRecordings.map((recording) => (
                    <tr key={recording.filename}>
                        <td>{recording.participant}</td>
                        <td>{recording.filename}</td>
                        <td>{recording.comment}</td>
                    </tr>
                ))}
            </tbody>
        </Table>
    );
};

export default PriorRecordingsTable;