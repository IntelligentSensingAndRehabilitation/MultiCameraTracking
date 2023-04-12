import React, { useContext } from 'react';
import { Table } from 'react-bootstrap';
import Accordion from 'react-bootstrap/Accordion';
import { AcquisitionState } from "../AcquisitionApi";

const CameraStatusTable = ({ api }) => {

    const { cameraStatusList } = useContext(AcquisitionState);

    return (
        <Accordion defaultActiveKey="0" className="g-4 p-2">
            <Accordion.Item eventKey="0">
                <Accordion.Header>Camera Statuses</Accordion.Header>
                <Accordion.Body>
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
                </Accordion.Body>
            </Accordion.Item>
        </Accordion>

    );
};

export default CameraStatusTable;