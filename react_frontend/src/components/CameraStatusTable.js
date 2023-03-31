import { useContext, useState, useEffect } from 'react';
import axios from 'axios';
import { Table } from 'react-bootstrap';
import { AcquisitionState } from "../AcquistionApi";

const CameraStatusTable = ({ api }) => {

    const { cameraStatusList } = useContext(AcquisitionState);

    return (
        <Table id="camera_status_table" striped bordered hover>
            <thead>
                <tr>
                    <th>Serial Number</th>
                    <th>Status</th>
                    <th>Pixel Format</th>
                    <th>Binning Horizontal</th>
                    <th>Binning Vertical</th>
                    <th>Width</th>
                    <th>Height</th>
                    <th>Sync Offset</th>
                </tr>
            </thead>
            <tbody>
                {cameraStatusList.map((cameraStatus) => (
                    <tr key={cameraStatus.SerialNumber}>
                        <td>{cameraStatus.SerialNumber}</td>
                        <td>{cameraStatus.Status}</td>
                        <td>{cameraStatus.PixelFormat}</td>
                        <td>{cameraStatus.BinningHorizontal}</td>
                        <td>{cameraStatus.BinningVertical}</td>
                        <td>{cameraStatus.Width}</td>
                        <td>{cameraStatus.Height}</td>
                        <td>{cameraStatus.SyncOffset}</td>
                    </tr>
                ))}
            </tbody>
        </Table>
    );
};

export default CameraStatusTable;